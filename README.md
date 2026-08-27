# RAG Pipeline with Hybrid Search, Reranking, and Production Instrumentation

![Eval Suite](https://github.com/mikepod-dev/rag-pipeline/actions/workflows/eval.yml/badge.svg)

A retrieval-augmented generation system built from scratch in Python — hybrid search (BM25 + embeddings), Reciprocal Rank Fusion, cross-encoder reranking, an automated evaluation harness, and production instrumentation (cost tracking, logging, caching, latency measurement).

**Live monitoring dashboard:** [rag-pipeline.streamlit.app](https://rag-pipeline-s4mw8mbzrbbv8fzdm5lwgt.streamlit.app) — real cost/usage data from 489 logged queries across development and testing.

Stress-tested against 796 real chunks (Wikipedia + hand-written docs), which surfaced and led to fixing a genuine retrieval bug — not a synthetic exercise. Full writeup: [`CASE_STUDY.md`](./CASE_STUDY.md).

## What's in here

- **`pipeline.py`** — the core system: chunking (with overlap), embeddings, hybrid search + RRF, cross-encoder reranking, LLM generation with contradiction detection, caching, cost/latency tracking, input validation
- **`eval.py`** — automated evaluation harness: retrieval accuracy and answer accuracy scored separately, mixing deterministic keyword checks with majority-vote LLM-as-judge for genuinely subjective cases
- **`fetch_articles.py`** — pulls real Wikipedia articles to build a realistic, competitive document set
- **`docs/`** — the document set (hand-written + real Wikipedia content, including deliberately conflicting sources for testing contradiction handling)
- **`generate_split_brain_data.py`** — split-brain LoRA data generation: an independent clerk model audits every candidate a separate generator model produces, with a structural guard preventing the fine-tuning target from ever auditing its own output
- **`finalize_module1_dataset.py`** — real tokenizer-based length audit (actual Llama-3.1 tokenizer, not a word-count approximation) plus accept/quarantine dataset split

## Key findings (see full case study for details)

1. **A silent BM25 length-bias bug** caused an unrelated long document to outrank the correct short one — found via targeted score debugging, fixed with Reciprocal Rank Fusion instead of ad hoc weight tuning.
2. **The eval harness itself had two real bugs** — an overly strict retrieval check, and a dead field that silently fell back to old scoring logic. Both found by noticing behavior that didn't match expectations, not by inspecting code first.
3. **Semantic caching was built, tested, and rejected** after finding the safety margin between "same question, different words" (0.8455 cosine similarity) and "different question entirely" (0.8149) was too thin to threshold safely — a decision backed by data, not intuition.
4. **A split-brain generation/audit architecture, verified at 796-chunk scale** — an independent clerk model catches real fabrications a same-model self-check would likely miss, including a case where it correctly rejected a claim that appeared verbatim in the very output it was checking, confirming it verifies against source content rather than generator confidence.
5. **A misconfigured reasoning parameter was silently burning most of the generation budget** — hidden "thinking" tokens on a reasoning model inflated per-chunk cost 4x with no quality benefit on a task that needed none, found by logging token-level cost breakdowns rather than trusting the aggregate number. Fix cut projected full-corpus cost from $1.26 to $0.31.

## Stack

Python · ChromaDB · sentence-transformers (bi-encoder + cross-encoder) · rank-bm25 · OpenRouter (Claude Haiku)

## Running it

```bash
pip install -r requirements.txt
# create a .env file with OPENROUTER_API_KEY=your_key
python pipeline.py       # interactive Q&A
python eval.py           # run the automated eval suite
```

## Fine-tuning data prep (LoRA)

Split-brain pipeline for generating and auditing instruction-tuning data from the source corpus -- a generator model produces candidates, an independent clerk model audits them for faithfulness, and a deterministic Python gate (not an LLM-emitted score) decides accept vs. quarantine.

```bash
# requires OPENROUTER_API_KEY and HF_TOKEN (gated Llama-3.1 tokenizer access) in .env
python generate_split_brain_data.py 796
python finalize_module1_dataset.py synthetic_run_<timestamp>.jsonl
```

Full findings and real numbers: Findings 19-23 in [`CASE_STUDY.md`](./CASE_STUDY.md).
