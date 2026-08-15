# RAG Pipeline with Hybrid Search, Reranking, and Production Instrumentation

![Eval Suite](https://github.com/mikepod-dev/rag-pipeline/actions/workflows/eval.yml/badge.svg)

A retrieval-augmented generation system built from scratch in Python — hybrid search (BM25 + embeddings), Reciprocal Rank Fusion, cross-encoder reranking, an automated evaluation harness, and production instrumentation (cost tracking, logging, caching, latency measurement).

Stress-tested against 796 real chunks (Wikipedia + hand-written docs), which surfaced and led to fixing a genuine retrieval bug — not a synthetic exercise. Full writeup: [`CASE_STUDY.md`](./CASE_STUDY.md).

## What's in here

- **`pipeline.py`** — the core system: chunking (with overlap), embeddings, hybrid search + RRF, cross-encoder reranking, LLM generation with contradiction detection, caching, cost/latency tracking, input validation
- **`eval.py`** — automated evaluation harness: retrieval accuracy and answer accuracy scored separately, mixing deterministic keyword checks with majority-vote LLM-as-judge for genuinely subjective cases
- **`fetch_articles.py`** — pulls real Wikipedia articles to build a realistic, competitive document set
- **`docs/`** — the document set (hand-written + real Wikipedia content, including deliberately conflicting sources for testing contradiction handling)

## Key findings (see full case study for details)

1. **A silent BM25 length-bias bug** caused an unrelated long document to outrank the correct short one — found via targeted score debugging, fixed with Reciprocal Rank Fusion instead of ad hoc weight tuning.
2. **The eval harness itself had two real bugs** — an overly strict retrieval check, and a dead field that silently fell back to old scoring logic. Both found by noticing behavior that didn't match expectations, not by inspecting code first.
3. **Semantic caching was built, tested, and rejected** after finding the safety margin between "same question, different words" (0.8455 cosine similarity) and "different question entirely" (0.8149) was too thin to threshold safely — a decision backed by data, not intuition.

## Stack

Python · ChromaDB · sentence-transformers (bi-encoder + cross-encoder) · rank-bm25 · OpenRouter (Claude Haiku)

## Running it

```bash
pip install -r requirements.txt
# create a .env file with OPENROUTER_API_KEY=your_key
python pipeline.py       # interactive Q&A
python eval.py           # run the automated eval suite
```
