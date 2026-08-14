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

The embedding model correctly judged the coffee content as *less* relevant (lower vector score) — but the BM25 scores told a different story.

**Root cause, precisely:** BM25's length-normalization term (`|D| / avgdl`, weighted by `b=0.75`) is designed to penalize documents longer than the corpus average. Because the corpus was uniformly chunked into ~100-word segments with fixed overlap, every chunk's length sits close to the corpus-wide average by construction — collapsing `|D| / avgdl` to approximately 1 for nearly every candidate, regardless of source document. With the normalization term reduced to a near-constant across the index, it stops meaningfully differentiating between chunks, and raw term frequency (`f(q,D)`) becomes the dominant driver of score differences instead. This is visible directly in the debug output: individual `wiki_coffee.txt` chunks ranged from a raw BM25 score of 0.000 to 9.225 — a spread driven by term-frequency saturation within specific chunks, not by document length in the naive whole-article sense. My original weighting formula (`vector_score + bm25_score * 0.1`) let this term-frequency variance dominate the combined score.

**The fix:** replaced the ad hoc weighted-sum with **Reciprocal Rank Fusion** — combining *rank positions* from each retrieval method instead of raw, incomparably-scaled scores. This is the standard, robust technique for exactly this problem, and it eliminated the coffee-contamination failure entirely with no further hand-tuning of magic weight constants.

**What the two-stage architecture then proved:** even after RRF, the correct document (`animals_overview.txt`) was still ranked 18th in the fused candidate list — outside a naive top-2 cutoff. Because reranking operates on a wider pool (25 candidates) and reads the actual question against each candidate directly, the cross-encoder correctly promoted it back to #1. This validated the retrieve-wide-then-rerank design under a real failure case, not just in theory.

**A third, distinct retrieval bug surfaced on a different question — one relevant document flooding the candidate pool.** Asking "Do cats hunt in packs?" returned nothing useful: `wiki_cat.txt` alone accounted for roughly 19 of the top 25 RRF-ranked candidates, crowding out `animals_overview.txt` (the document that actually answers the question) entirely, even though it ranked 10th in pure semantic search alone. This wasn't the same bug as the coffee-contamination case — it was a *relevant* document dominating the pool through sheer chunk count, not an irrelevant one winning on inflated keyword scores.

**Fix:** added a per-source diversity cap to the retrieval stage — no single document can contribute more than 3 chunks to the candidate pool, regardless of how well its chunks individually score. This guarantees room for multiple genuinely relevant documents to reach the reranker instead of one large document monopolizing the results. After the fix, `animals_overview.txt` reached the candidate pool and the reranker correctly promoted it, resolving the case with a verified-correct final answer.

**Validated with a parameter sweep, not left as an untested guess.** I swept `max_per_source` across values 1 through 5, plus an uncapped control (999, reproducing the original pre-fix condition), against the eval cases with a known expected source. Every capped value (1-5) held at 100% retrieval accuracy, while the uncapped control dropped to 80% — confirming the cap mechanism itself is what matters, not a specific value. At this sample size (5 cases), 1 through 5 are statistically indistinguishable from each other, so I'm not claiming 3 is uniquely optimal — only that capping the pool at all is empirically justified, and 3 was a reasonable choice within the range that performed equivalently. A larger eval set would be needed to meaningfully distinguish between cap values themselves.

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

**Honest scoping decision:** proving genuine multi-hop reasoning would require either running full extraction across all 796 chunks (796 real LLM calls — meaningful, deliberate cost) or a smaller, targeted sample specifically including a document like `wiki_wolf.txt` where "wolves" would plausibly appear as a subject. Given this was a late-session proof-of-concept, I initially validated only the *mechanism* on an untargeted 15-chunk sample.

**Follow-up: scaled and targeted the sample, found a second sampling bug, then got a real proof of multi-hop reasoning.** Expanding to a 60-chunk sample explicitly listing `wiki_wolf.txt` as a source still produced zero wolf-related edges — investigation showed the sampling code took the *first* 60 matching chunks in file order, and since `wiki_dog.txt` alone contains dozens of chunks, the cutoff was exhausted before ever reaching `wiki_wolf.txt`. This is the same class of bug as the BM25 length-bias and single-document pool-dominance issues found earlier: one large source silently crowding out another, this time in sampling rather than retrieval. **Fixed by sampling proportionally per source** (10 chunks from each target document, rather than the first N overall) — conceptually the same diversity-cap fix already applied to retrieval, now applied to graph-building.

With wolf-related content genuinely included, the graph produced real, verified multi-hop chains that plain retrieval cannot produce, including a genuine 3-hop connection spanning three separate entities across different source chunks:

```
dogs --[can_communicate_with]--> humans --[domesticate]--> sheep --[provide]--> meat
```

No single document chunk states a direct relationship between dogs and sheep — the graph assembled this connection automatically by traversing shared intermediate entities across chunks originating from different documents. This is the concrete capability gap Graph RAG exists to close, demonstrated with a real, inspectable result rather than left as an unproven claim.

**Remaining known limitation:** singular/plural and phrasing variants of the same entity (e.g., "dog" vs. "dogs," "domesticated_from" vs. "domesticated from") are not yet unified, meaning some genuinely-equivalent nodes are still tracked separately. A production version would need a real entity-resolution layer (e.g., embedding-based similarity merging) rather than simple lowercase-and-strip normalization.

**What a full production version would still require:** full-corpus extraction across all 796 chunks, the entity-resolution improvement above, and a hybrid query router that sends relationship-style questions to the graph and factual-lookup questions to the existing hybrid search + reranking pipeline.

---

## Finding 8: Calibrating the judge against a human baseline — and finding it fails without an anchor

Every eval case up to this point relied on either deterministic keyword checks or an LLM judge compared against a human-written `expected_answer` reference. That reference does real work: it gives the judge something concrete to check against. I hadn't tested what happens when that anchor is removed — i.e., whether the judge can independently assess answer quality on its own, the way it would need to on genuinely novel production traffic with no pre-written reference answer.

**Method:** generated 20 new questions against the live system — not the original 11 eval cases, to avoid biasing my own grading with prior knowledge of expected answers. Graded all 20 blind, myself, based purely on reading each question and answer. Independently ran the same 20 through a 3-way majority-vote judge (structurally identical to the one used in `eval.py`), with **no `expected_answer` field provided** — question and answer only, replicating a real production condition where no reference exists yet.

**First run of the judge was invalid, and worth documenting why.** The initial implementation routed the grading prompt through the existing `ask_llm` function, which has a system-prompt template hard-coded around "answer using ONLY the following context." With an empty context list, the model interpreted the entire grading task through that lens and returned "the provided context is empty, I cannot verify this" for the majority of cases — a real structural bug: reusing a function outside the specific task it was designed for, without checking whether its embedded framing still applied. Rebuilt as a fully isolated grading call, bypassing that prompt template entirely, to get a valid result.

**The result, once valid:**

| Metric | Value |
|---|---|
| Raw agreement (human vs. judge) | 14/20 (70.0%) |
| Human PASS rate | 85.0% |
| Judge PASS rate | 85.0% |
| **Cohen's kappa** | **-0.176** |

Cohen's kappa corrects raw agreement for the rate you'd expect by chance alone, given each grader's overall pass rate — necessary here, since both graders independently passed 85% of cases, which would inflate raw agreement even if the two were tracking completely different things. A kappa near 0 means agreement no better than chance; a negative kappa means **agreement worse than chance** — the judge's calls were, in aggregate, actively uncorrelated with (arguably anti-correlated with) independent human judgment on this unanchored question set.

**Inspecting the 6 disagreements showed a real, interpretable pattern, not random noise:**
- The judge passed several answers I failed for being technically-present-but-thin or resting on an unverified causal claim (e.g., "cats sleep more as they age" stated as settled fact).
- The judge failed several answers I passed that were, on inspection, clearly correct and well-presented (a full comparison table correctly contrasting two company policies; a correctly-stated product code and unit count).
- One case (coffee's continent of origin) the judge had itself flagged as technically imprecise in the earlier, invalid run ("Arabia is a peninsula, not a continent") — but reversed to PASS once run without an anchor, suggesting the earlier catch was closer to an artifact of the broken prompt than genuine reasoning.

**Honest conclusion:** the LLM-as-judge technique used throughout this project's eval harness performs well when anchored against a human-written reference answer, but **that reliability appears to come substantially from the anchor itself, not from the judge's independent reasoning.** Without a reference to check against — the exact condition real production traffic would present — agreement with a careful human reader was no better than chance. This means the eval harness's demonstrated 11/11 accuracy should be read as *validated against known reference answers*, not as evidence the judging mechanism would reliably self-assess novel, un-anchored production queries. A real deployment would need either a continuously human-curated reference set, a materially stronger judge model, or a different evaluation strategy entirely for genuinely novel traffic — and this gap should be disclosed as a known limitation, not implied away by a clean eval score.

---

## Finding 9: Containerizing for deployment — a real free-tier memory ceiling

To move from a local script to a genuinely deployable service, I wrapped the pipeline in a FastAPI endpoint (`POST /ask`), containerized it with Docker, and verified it locally: the containerized instance correctly connected to the live Qdrant Cloud cluster and answered questions correctly through a real HTTP request from outside the container — a working, portable, self-contained service, not just a local script.

**Deploying to Render's free tier failed at runtime, not build time — a distinct and important difference.** The Docker build itself completed successfully (dependencies installed, image exported and pushed without error). The failure occurred when the container actually started and attempted to load its models into memory:

```
Out of memory (used over 512Mi)
```

**Root cause, precisely:** Render's free tier caps container memory at 512MB. The container loads two transformer models directly into memory on startup — the `sentence-transformers` bi-encoder for embeddings and the `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker — plus PyTorch itself as the underlying runtime, whose baseline memory footprint alone is substantial before any model weights are even loaded. This combination reliably exceeds 512MB, which is why the build succeeds (no memory pressure during dependency installation) while the runtime startup fails (both models loading simultaneously into a hard-capped container).

**Decision, consistent with the project's other free-tier findings (Qdrant's 1GB storage cap, the batch-upload timeout):** rather than upgrade to a paid tier to make the constraint disappear, I documented the actual limitation with the actual evidence, understood precisely why it occurs, and evaluated it as an engineering tradeoff rather than a blocker. A production deployment of this exact architecture would require one of: a smaller embedding model, lazy-loading the reranker only for requests that need it rather than at startup, splitting embedding/reranking into a separate service from the API layer, or simply provisioning adequate memory (Render's Standard tier, 2GB, would comfortably fit both models with room to spare). The free tier's 512MB ceiling is a real, disclosed constraint of this specific deployment choice, not a flaw in the architecture itself — the same pipeline runs correctly with no memory issues on hardware with adequate RAM, as proven by both the local machine and local Docker container tests before deployment.

**A follow-up architectural option was evaluated and found to have its own disclosed boundary: a fully "stateless" API server is not achievable at $0.** The idea: eliminate local memory pressure entirely by moving BM25 keyword search and embedding computation out of the API server and into Qdrant Cloud's native hybrid search and server-side inference features, so the Render container would hold no models at all — just pass questions to Qdrant and receive fully-ranked results.

Checked against Qdrant's actual documentation before building anything, rather than assuming it would work: **Qdrant's server-side inference (`cloud_inference=True`) — the specific feature that lets Qdrant compute embeddings from raw text server-side, removing the need to run `sentence-transformers` locally — explicitly requires a dedicated paid cluster.** There is no free-tier path to eliminating local embedding computation entirely; the computation has to happen somewhere, and Qdrant's free tier doesn't offer to do it for you.

**Scoped, honest revision of the idea that stays within the $0 constraint:** migrate BM25 keyword search from local Python (`rank-bm25`, local tokenization, an in-memory index rebuilt every startup) to Qdrant's native sparse vectors, which run server-side and are available on the free tier. This is a real, genuine architecture improvement — removing an entire local index and its rebuild cost — while keeping `sentence-transformers` running locally for query embedding, since no free path exists to avoid that specific computation. The reranker's memory footprint remains a separate, already-documented constraint. This is disclosed here as a deliberate scope boundary decided *before* implementation, not a limitation discovered after the fact.

**Implemented and rigorously verified.** Created a new Qdrant collection (`rag_docs_hybrid`) configured with both a named dense vector field and a named sparse vector field, so a single point can carry both simultaneously. Each chunk's sparse vector is now computed server-side by Qdrant's built-in `Qdrant/bm25` model at upload time — no local tokenization, no local `BM25Okapi` index construction, no local index rebuild on every startup. Retrieval now uses Qdrant's native `Prefetch` + `FusionQuery(fusion=Fusion.RRF)` API: two parallel server-side searches (dense and sparse), fused with RRF *inside Qdrant itself* — the same algorithm hand-implemented earlier in this project (Finding 2), now running natively in the cloud instead of in local Python.

Verified against the full 11-case evaluation suite, not a single spot-check: **11/11 retrieval accuracy, 11/11 answer accuracy — identical to the local-BM25 architecture**, confirming the migration introduced no regression while removing a meaningful chunk of local computation and memory (the entire BM25 index and tokenized corpus no longer need to live in the API server's memory). This directly reduces the pressure that caused the free-tier memory failure documented above, while remaining honest that the reranker and the dense embedding model are still local, disclosed constraints rather than problems papered over.

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
- Making a deliberate, disclosed scoping decision (validating a mechanism on a sample vs. proving a capability at full scale), then following through to actually prove the capability once time allowed — including finding and fixing a second, related sampling bug (one large source crowding out a smaller one) using the same diversity-cap principle already applied to retrieval

---

## Stack

Python · ChromaDB · `sentence-transformers` (bi-encoder + cross-encoder) · `rank-bm25` · OpenRouter (Claude Haiku) · git
