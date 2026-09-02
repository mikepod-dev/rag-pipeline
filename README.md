# RAG Pipeline with Hybrid Search, Reranking, and Production Instrumentation

![Eval Suite](https://github.com/mikepod-dev/rag-pipeline/actions/workflows/eval.yml/badge.svg)

A retrieval-augmented generation system built from scratch in Python — hybrid search (BM25 + embeddings), Reciprocal Rank Fusion, cross-encoder reranking, an automated evaluation harness, and production instrumentation — then extended through a self-directed MLOps curriculum covering LoRA fine-tuning, RAFT-style robustness training, automated evaluation gating, and embedding-space safety checks.

**Live monitoring dashboard:** [rag-pipeline.streamlit.app](https://rag-pipeline-s4mw8mbzrbbv8fzdm5lwgt.streamlit.app) — real cost/usage data from logged production queries.

Every stage was stress-tested against a real 796-chunk corpus, not a synthetic exercise — this surfaced and led to fixing genuine bugs at every layer, from a BM25 scoring formula to a Rust regex engine's undocumented limitations to a stale Windows environment variable. Full writeup, 34 documented findings: [`CASE_STUDY.md`](./CASE_STUDY.md).

## By the Numbers

| Metric | Value |
|---|---|
| Final retrieval / answer accuracy | 11/11 · 11/11 |
| Unit test runtime reduction | 50.33s → 0.05s (~1,000×) |
| Judge kappa vs. human baseline, unanchored | −0.176 (worse than chance) → 0.138 (two independent redesigns) |
| Generation cost reduction from one parameter fix | $1.26 → $0.31 per 796-chunk run (75%) |
| False-abstain rate before/after targeted retrain | 11.3% → 1.3% |
| Pairwise embedding-similarity comparisons computed | 634,410 pairs in under 10ms |
| Flawed automated corrections caught by human review, first real use | 3 of 3 |

## Curriculum modules completed

- **Module 1 — Data pipeline + LoRA fine-tuning:** split-brain generation/validation architecture, real GPU training with genuine infrastructure failures worked through
- **Module 2 — RAFT (distractor-robustness training):** trained, measured, and honestly reported a null result against the project's own hypothesis
- **Module 3 — Automated evaluation gate:** faithfulness scoring, teacher-model correction, circuit breaker, and a human-in-the-loop staging step that caught its own pipeline's flaws in first real use
- **Module 4 — Embedding geometry watchdog:** canary-verified corpus safety check with a real O(n^2) scaling analysis

## What's in here

**Core pipeline**
- **`pipeline.py`** — the core system: chunking (with overlap), embeddings, hybrid search + RRF, cross-encoder reranking, LLM generation with contradiction detection, caching, cost/latency tracking, input validation
- **`eval.py`** — automated evaluation harness: retrieval accuracy and answer accuracy scored separately, mixing deterministic keyword checks with majority-vote LLM-as-judge for genuinely subjective cases
- **`fetch_articles.py`** — pulls real Wikipedia articles to build a realistic, competitive document set
- **`docs/`** — the document set (hand-written + real Wikipedia content, including deliberately conflicting sources for testing contradiction handling)

**Fine-tuning data pipeline (Module 1)**
- **`generate_split_brain_data.py`** — split-brain LoRA data generation: an independent clerk model audits every candidate a separate generator model produces, with a structural guard preventing the fine-tuning target from ever auditing its own output
- **`finalize_module1_dataset.py`** — real tokenizer-based length audit (actual Llama-3.1 tokenizer, not a word-count approximation) plus accept/quarantine dataset split

**RAFT: distractor-robustness training (Module 2)**
- **`generate_raft_records.py`** — builds distractor-injected training records with band-based similarity search against the full corpus, plus explicit abstain-training examples
- **`compare_raft_vs_baseline.py`** — paired faithfulness comparison between a RAFT-trained adapter and the Module 1 baseline, using the same split-brain clerk methodology

**Automated evaluation gate (Module 3)**
- **`ragas_gate.py`** — deterministic claim-based faithfulness scoring (not an LLM-emitted float), teacher-model correction for failing answers, and a combined failure-rate/cost circuit breaker
- **`review_pending_corrections.py`** — the human-in-the-loop approval step: no automated correction reaches a trusted dataset without explicit review

**Embedding geometry watchdog (Module 4)**
- **`geometry_watchdog_explore.py`** — computes the full pairwise similarity distribution to empirically choose a clumping threshold for this specific corpus/embedder
- **`geometry_watchdog.py`** — the production check: canary-verified detection, machine-readable report, non-zero exit code for CI gating

## Key findings (see full case study for all 34)

1. **A silent BM25 length-bias bug** caused an unrelated long document to outrank the correct short one — found via targeted score debugging, fixed with Reciprocal Rank Fusion instead of ad hoc weight tuning.
2. **Semantic caching was built, tested, and rejected** after finding the safety margin between "same question, different words" (0.8455 cosine similarity) and "different question entirely" (0.8149) was too thin to threshold safely — a decision backed by data, not intuition.
3. **An LLM-as-judge, tested without a reference answer to anchor it, agreed with a human grader worse than chance** (Cohen's kappa −0.176) — leading to three successive judge redesigns, the best of which matched a standard library's real-world agreement level via a structurally different, single-call mechanism.
4. **A split-brain generation/audit architecture, verified at full 796-chunk scale**, empirically catches real fabrications a same-model self-check would likely miss — including a case where it correctly rejected a claim that appeared verbatim in the very output it was checking.
5. **A misconfigured reasoning parameter was silently burning most of the generation budget** — hidden "thinking" tokens on a reasoning model inflated per-chunk cost 4x with no quality benefit on a task that needed none. Fix cut projected full-corpus cost from $1.26 to $0.31.
6. **A RAFT-trained adapter's best-scoring checkpoint by loss produced a complete topic hallucination** on a directly-trained example — proving teacher-forced eval loss and real generation quality are not the same thing, and that a checkpoint has to be tested by actually reading its output.
7. **A trained model crossed the curriculum's own explicit >10% false-abstain failure threshold** on held-out data (11.3%); a targeted retrain, measured with the same rigor used to find the problem, cut it to 1.3%.
8. **RAFT training was measured against its own baseline and honestly reported as showing no advantage** (0.796 vs. 0.808 mean faithfulness) — including catching that an apparent baseline advantage was actually the complete absence of a capability the baseline was never trained on, not evidence of quality.
9. **A human-in-the-loop safety gate, built in direct response to a real automated failure, caught 3 of 3 flawed corrections** in its first real use — including a second, previously undiscovered failure mode where a "smarter" correction model introduced true-but-ungrounded claims.
10. **An embedding-geometry watchdog's "zero bugs found" result was proven meaningful, not just unverified**, by injecting a known-duplicate canary before trusting the real corpus's results — and the O(n^2) scaling analysis found memory, not compute time, is the actual wall for this approach at scale.

## Stack

Python · Qdrant (hybrid dense + server-side sparse vectors) · `sentence-transformers` (bi-encoder + cross-encoder) · FastAPI · Celery + Redis · Unsloth/QLoRA · OpenRouter (DeepSeek, GPT-4o-mini, Gemini Flash, Claude Sonnet) · git

## Running it

```bash
pip install -r requirements.txt
# create a .env file with OPENROUTER_API_KEY=your_key
python pipeline.py       # interactive Q&A
python eval.py           # run the automated eval suite
```

## Fine-tuning data prep (LoRA)

Split-brain pipeline for generating and auditing instruction-tuning data from the source corpus — a generator model produces candidates, an independent clerk model audits them for faithfulness, and a deterministic Python gate (not an LLM-emitted score) decides accept vs. quarantine.

```bash
# requires OPENROUTER_API_KEY and HF_TOKEN (gated Llama-3.1 tokenizer access) in .env
python generate_split_brain_data.py 796
python finalize_module1_dataset.py synthetic_run_<timestamp>.jsonl
```

Full findings and real numbers: Findings 19-23 in [`CASE_STUDY.md`](./CASE_STUDY.md).

## RAFT distractor-robustness training

Generates training records with band-based distractor injection (low/mid/high similarity to the query) and explicit abstain examples, then compares a RAFT-trained adapter against the Module 1 baseline using the same split-brain clerk methodology.

```bash
python generate_raft_records.py --abstain-fraction 0.05 --output raft_records.jsonl
# training itself runs in Colab (Unsloth/QLoRA) -- see CASE_STUDY.md Findings 24-27 for the recipe
python compare_raft_vs_baseline.py
```

Full findings and real numbers: Findings 28-32 in [`CASE_STUDY.md`](./CASE_STUDY.md).

## Automated evaluation gate + human review

Gates real production answers on deterministic claim-based faithfulness, escalates failures to a teacher model for correction, and requires explicit human approval before anything reaches a trusted dataset.

```bash
# requires OPENROUTER_API_KEY in .env
python ragas_gate.py --limit 25 --cost-threshold 1.00
python review_pending_corrections.py --version 1
```

Full findings and real numbers: Finding 33 in [`CASE_STUDY.md`](./CASE_STUDY.md).

## Embedding geometry watchdog

Computes the full pairwise cosine-similarity matrix against the real corpus, flags cross-document clumping above an empirically-chosen threshold, and proves its own detection mechanism works via an injected canary before trusting any result.

```bash
python geometry_watchdog_explore.py      # see the real similarity distribution first
python geometry_watchdog.py --threshold 0.80
```

Full findings and real numbers: Finding 34 in [`CASE_STUDY.md`](./CASE_STUDY.md).
