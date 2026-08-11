# Building and Debugging a Production-Minded RAG Pipeline
### A case study in retrieval failures, evaluation discipline, and knowing when to reject your own feature

---

## Summary

I built a retrieval-augmented generation (RAG) system from scratch — hybrid search (BM25 + embeddings), Reciprocal Rank Fusion, cross-encoder reranking, an automated evaluation harness, cost/latency tracking, and caching — then deliberately stress-tested it against a 796-chunk real-world document set. That stress test surfaced a genuine, non-obvious retrieval bug that had been silently possible from the start but only became visible at realistic scale. I also built, rigorously tested, and ultimately **rejected** a semantic caching feature after finding it introduced a silent wrong-answer risk — a decision backed by hard similarity-score evidence, not intuition.

This writeup documents the real engineering arc: what broke, how I diagnosed it, what I measured, and what I chose not to ship.

---

## Architecture

- **Chunking:** word-based with configurable overlap (30 words) to prevent facts from being split across chunk boundaries
- **Retrieval:** hybrid search combining dense embeddings (`all-MiniLM-L6-v2`) and BM25 keyword search, fused via **Reciprocal Rank Fusion (RRF)**
- **Reranking:** a cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-scores a wide candidate pool (25) down to the final top-N, correcting cases where first-stage fusion alone misranks results
- **Generation:** Claude Haiku via OpenRouter, with an explicit instruction to flag contradictions across sources rather than silently resolving them
- **Evaluation:** an automated harness scoring retrieval accuracy and answer accuracy separately — deterministic keyword checks where possible, majority-vote LLM-as-judge only where genuinely subjective
- **Production instrumentation:** per-call cost tracking, persistent JSONL query logging, exact-match answer caching, retrieval/generation latency splits, and input validation

---

## Finding 1: A demo that looked perfect (11/11) was never actually tested

The system passed 11/11 on its original evaluation set — three tiny, hand-written documents about dogs, cats, and coffee. That number meant almost nothing. With so few chunks, there was no real retrieval competition; almost any reasonable method would have scored well.

**The fix wasn't code — it was scale.** I pulled 8 real Wikipedia articles (~350,000 characters, chunked into 796 pieces) on topics deliberately chosen to overlap with the existing content — `Dog`, `Wolf`, `Domestication`, `Coffee`, `Caffeine` — specifically to create genuine competition between similar-sounding chunks. This is the condition under which retrieval quality actually gets tested.

Retrieval accuracy immediately dropped from 11/11 to 8/11 under real competitive pressure.

---

## Finding 2: A critical, silent BM25 length-bias bug

One failure stood out: asking **"What makes an animal a mammal?"** returned content from `wiki_coffee.txt` — a completely unrelated document.

I didn't accept the surface symptom. I added targeted debug instrumentation to print the raw vector score, raw BM25 score, and combined score for every candidate chunk, isolating the two competing documents (`animals_overview.txt`, the correct source, vs. `wiki_coffee.txt`, the incorrect one).

**The evidence:**

| Document | Vector score | Raw BM25 score | Combined score |
|---|---|---|---|
| `animals_overview.txt` (correct) | 0.531 | 2.450 | **0.776** |
| `wiki_coffee.txt` chunk (wrong) | 0.364 | 9.225 | **1.286** |

The embedding model correctly judged the coffee content as *less* relevant (lower vector score) — but raw BM25 scores are unbounded and reward term frequency across a document's full length. A 57,000-character article racks up large cumulative BM25 scores on generic words alone, regardless of topical relevance. My original weighting formula (`vector_score + bm25_score * 0.1`) let this length bias dominate.

**The fix:** replaced the ad hoc weighted-sum with **Reciprocal Rank Fusion** — combining *rank positions* from each retrieval method instead of raw, incomparably-scaled scores. This is the standard, robust technique for exactly this problem, and it eliminated the coffee-contamination failure entirely with no further hand-tuning of magic weight constants.

**What the two-stage architecture then proved:** even after RRF, the correct document (`animals_overview.txt`) was still ranked 18th in the fused candidate list — outside a naive top-2 cutoff. Because reranking operates on a wider pool (25 candidates) and reads the actual question against each candidate directly, the cross-encoder correctly promoted it back to #1. This validated the retrieve-wide-then-rerank design under a real failure case, not just in theory.

**A third, distinct retrieval bug surfaced on a different question — one relevant document flooding the candidate pool.** Asking "Do cats hunt in packs?" returned nothing useful: `wiki_cat.txt` alone accounted for roughly 19 of the top 25 RRF-ranked candidates, crowding out `animals_overview.txt` (the document that actually answers the question) entirely, even though it ranked 10th in pure semantic search alone. This wasn't the same bug as the coffee-contamination case — it was a *relevant* document dominating the pool through sheer chunk count, not an irrelevant one winning on inflated keyword scores.

**Fix:** added a per-source diversity cap to the retrieval stage — no single document can contribute more than 3 chunks to the candidate pool, regardless of how well its chunks individually score. This guarantees room for multiple genuinely relevant documents to reach the reranker instead of one large document monopolizing the results. After the fix, `animals_overview.txt` reached the candidate pool and the reranker correctly promoted it, resolving the case with a verified-correct final answer.

**Final evaluation result: 11/11 retrieval accuracy, 11/11 answer accuracy** — every case either independently verified correct, or correctly scored as source-agnostic where multiple valid documents legitimately contain the same fact.

---

## Finding 3: The eval harness itself needed hardening — twice

Two separate flaws were found *in the evaluation tooling*, not the RAG system:

1. **Retrieval scoring was stricter than what actually mattered.** The check only verified the #1 retrieved result matched expectations, but the system actually used the top-2 results for generation. A "failing" case turned out to be a correct answer built from a top-2 chunk that just wasn't ranked #1. Fixed by checking whether the expected source appeared *anywhere* in the retrieved set, matching what the LLM actually saw.

2. **LLM-as-judge was measurably inconsistent.** Running the identical eval twice, with no code changes, produced different pass/fail verdicts on the same answers — including one case where the judge's own stated reasoning contradicted its verdict. Rather than trust a single noisy LLM call, I converted the majority of subjective checks into deterministic keyword-based assertions, and added 3-call majority voting for the cases that genuinely couldn't be reduced to keyword matching.

**Net result:** 8 of 11 final eval cases are fully deterministic — instant, free, perfectly reproducible — with only 1 case still relying on (now majority-voted) LLM judgment.

---

## Finding 4: Catching a bug I introduced myself

While hardening the eval harness, I instructed a new `must_contain_any` field be added to support multiple acceptable answer phrasings — but the scoring logic that was supposed to *read* that field was never updated to check for it. The result: the eval silently fell back to the old LLM-judge path with no error, no warning, and no visible sign anything was wrong.

This was caught not by inspecting the code, but by noticing the eval's pass/fail behavior didn't match what the newly-added rule should have produced — then tracing the discrepancy back to its source. It's a concrete example of the exact failure mode that makes AI-assisted development risky if unverified: a plausible-looking instruction can silently do nothing.

---

## Finding 5: Building, testing, and rejecting semantic caching

Exact-match caching (question text → answer) was extended to **semantic caching** — embedding each question and matching against previously-cached questions by cosine similarity, so paraphrased repeats could also hit the cache.

I tested this rigorously rather than assuming it worked:

| Question pair | Relationship | Cosine similarity |
|---|---|---|
| "How many hours do cats sleep?" vs. "What's the typical amount of daily sleep for a cat?" | Same meaning, different words | **0.8455** |
| "How many hours do cats sleep?" vs. "How many hours do dogs sleep?" | Different topic entirely | **0.8149** |

The gap between "genuinely the same question" and "genuinely a different question" was only **0.03** — far too thin a margin to safely threshold. At a permissive setting (0.80), the dogs question returned the cats answer verbatim, with full confidence and no indication of error. At a safe setting (0.93+), the feature never fired on real paraphrases at all, providing no benefit.

**Decision: rejected and removed**, with the finding documented rather than silently ignored. A feature that can't clear a safety bar shouldn't ship, even if it "mostly" works — and the reasoning behind that call, backed by concrete similarity scores, is itself the more valuable artifact than the feature would have been.

---

## Finding 6: Agentic RAG — a self-correcting loop, and a real limit of self-grading

Building on the plain/hybrid retrieval pipeline, I implemented an agentic loop: retrieve → generate → self-grade the answer → if insufficient, rewrite the query and retry (up to 2 additional attempts). This is a meaningfully different architecture from one-shot RAG — the system evaluates its own output and can actively recover from a bad first attempt rather than returning it directly.

**First iteration of the self-grader failed immediately.** Testing against a deliberately colloquial, oddly-phrased question ("What's the weird theory about bugs inside animals causing them to become pets?" — referring to a real hypothesis in the source material, the parasite-mediated domestication theory), the grader marked a clear non-answer ("I cannot answer this question... does not contain any theory") as `Sufficient: True`. The grader was judging the *surface shape* of the answer (does it look like a refusal) rather than the *cause* — it had no way to distinguish a correct refusal (information genuinely absent) from an incorrect one (retrieval simply grabbed the wrong content), because it never saw what was actually retrieved.

**Fix, attempt one:** gave the grader visibility into the retrieved context alongside the answer, and asked it to judge whether the retrieved content looked topically related to the question before accepting a refusal as valid.

**This fix revealed a second, more subtle limitation.** On retry, retrieval pulled genuinely topically-adjacent content (theories of *why humans keep pets* — evolutionary advantage, empathy side-effects) instead of the actually-correct source (the parasite-mediated domestication hypothesis, in a different document). The answer correctly reported this content didn't match and correctly refused — and the grader, now able to see the retrieved context, judged it as a legitimate, good-faith refusal, since the content genuinely was on-topic. **A near-miss that retrieves the right general subject but the wrong specific document is functionally indistinguishable, to a self-grader, from a case where the information is genuinely absent** — both present as "reasonable-looking context, correct-sounding refusal."

**Honest conclusion:** self-grading meaningfully improves on never checking at all, and successfully catches obviously bad retrieval. But it cannot reliably solve the harder case — a plausible-but-wrong retrieval — without a fundamentally different signal than "does this look reasonable," since a near-miss and a genuine absence can look identical from the answer's perspective. This is documented as a known, real limitation of the technique as implemented, not silently smoothed over.

---

## Finding 7: Graph RAG — a validated proof of concept, with a scope decision made explicit

To address multi-hop relationship questions (e.g., "what animals are descended from wolves") that plain vector/keyword retrieval structurally cannot answer well — since it retrieves isolated chunks with no concept of how entities relate to each other — I built a graph-based retrieval layer using `networkx`: an LLM extracts `(subject, relationship, object)` triples from document chunks, which are assembled into a directed graph that can be traversed for multi-hop connections.

**Built and validated on a 15-chunk sample** (not the full 796-chunk set — a deliberate scoping decision, explained below):

- Extraction worked: 15 chunks yielded 133 distinct entities and 112 relationships, correctly capturing real facts (`dogs --[descended_from]--> wolves`, `mammals --[characterized_by]--> hair or fur`)
- **Found and fixed a real entity-normalization bug**: the same real-world entity ("Dogs" vs. "dogs" vs. "Domesticated Mammals") was initially extracted with inconsistent capitalization, creating duplicate nodes for the same concept. A query for `graph.out_degree('Domesticated Mammals')` silently returned an empty list rather than erroring — the node didn't exist under that exact casing, and the failure was invisible rather than crashing. Fixed by lowercasing and stripping all entity names before adding them to the graph.
- Single-hop traversal (`dogs --[X]--> Y`) worked correctly and returned real, correct connections.
- **Tested for 2-hop traversal and found none — correctly, not as a bug.** Neither "wolves" nor "domesticated mammals" had any outgoing edges in the 15-chunk sample, meaning no entity appeared as *both* the target of one relationship and the source of another within that sample. This was verified directly (`out_degree` genuinely returning `0` for a confirmed-existing node) rather than assumed.

**Honest scoping decision:** proving genuine multi-hop reasoning would require either running full extraction across all 796 chunks (796 real LLM calls — meaningful, deliberate cost) or a smaller, targeted sample specifically including a document like `wiki_wolf.txt` where "wolves" would plausibly appear as a subject. Given this was a late-session proof-of-concept, I chose to validate the *mechanism* (extraction, normalization, traversal all work correctly) rather than scale to prove the *specific capability* (multi-hop answers) — a deliberate, documented scope boundary rather than an unstated limitation.

**What a production version would require:** full-corpus extraction, a more robust entity-resolution step (exact-match lowercasing is a first pass; real systems typically need embedding-based entity linking to merge near-duplicates like "Dog" and "Canis familiaris"), and a hybrid query router that sends relationship-style questions to the graph and factual-lookup questions to the existing hybrid search + reranking pipeline.

---

## What this project demonstrates



- Diagnosing failures by separating retrieval correctness from generation correctness, rather than treating "wrong answer" as one undifferentiated category
- Distinguishing a real regression from a stale test expectation — several apparent "failures" were the system correctly using better source material than the eval was written against
- Applying an industry-standard fix (RRF) instead of continuing to hand-tune an ad hoc formula once the root cause was understood
- Validating a two-stage retrieval architecture with a real failure case, not just a synthetic one
- Identifying and distinguishing between multiple distinct failure modes in the same retrieval pipeline (irrelevant-document dominance vs. relevant-document over-representation) rather than treating every retrieval miss as the same class of bug
- Recognizing that LLM-as-judge is itself a probabilistic, sometimes-inconsistent system requiring the same skepticism applied to any other component
- Making — and being able to justify with data — the decision *not* to ship a feature that introduced a silent-failure risk
- Recognizing when an added safeguard (self-grading) solves one failure mode but not a harder, structurally different one, and documenting that boundary explicitly rather than presenting a partial fix as complete
- Making a deliberate, disclosed scoping decision (validating a mechanism on a sample vs. proving a capability at full scale) rather than either overclaiming results or silently limiting scope without saying so

---

## Stack

Python · ChromaDB · `sentence-transformers` (bi-encoder + cross-encoder) · `rank-bm25` · OpenRouter (Claude Haiku) · git
