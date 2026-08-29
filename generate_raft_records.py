"""
generate_raft_records.py

Module 2 (RAFT): builds distractor-injected training records from accepted_dataset.jsonl.

Design decisions:
- Distractors are searched by cosine similarity to the QUERY (candidate.instruction),
  matching the curriculum spec exactly -- NOT similarity to the golden chunk.
- The golden chunk's own ID is hard-excluded from every distractor search. A real
  similarity-distribution check against 50 real records found golden chunks cluster
  at 0.5-0.85 similarity to their own query (median 0.71), directly overlapping the
  "high" band -- without explicit exclusion, the golden chunk could be returned as
  its own distractor.
- Similarity is computed against the FULL 796-chunk corpus locally (loaded once,
  reused across all queries), NOT via Qdrant top-K search. A 10-record test run
  found Qdrant's top-50 search consistently failed to find genuine "low" (~0.3)
  similarity chunks -- top-K is inherently biased toward the highest-similarity end
  of the distribution, and with only 796 total chunks, a true low-similarity match
  may not exist anywhere in a top-50 window at all. At this corpus size, computing
  full cosine similarity locally is cheap and removes the bias entirely.
- Three similarity bands per record: low (~0.3), mid (~0.6), high (~0.8) -- one
  distractor pulled from each band, giving N=3 distractors per RAFT record (the
  curriculum's stated range is 3-4; starting at 3, the minimum of that range).
- A configurable fraction of records are converted to "abstain" records: the golden
  chunk is entirely removed (distractors only), and the target answer is replaced
  with an explicit refusal string, per the curriculum's abstain-training requirement.

Usage:
    python generate_raft_records.py [--abstain-fraction 0.1] [--limit N]
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv(override=True)

COLLECTION = "rag_docs_hybrid"
BANDS = {"low": 0.3, "mid": 0.6, "high": 0.8}
BAND_TOLERANCE = 0.08  # accept a hit within +/- this of the target similarity
ABSTAIN_ANSWER = "The provided context does not contain this information."
SEED = 3407


def load_records(path: Path, limit: int | None) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    if limit:
        records = records[:limit]
    return records


def load_full_corpus(client: QdrantClient) -> tuple[list[str], np.ndarray, dict[str, dict]]:
    """
    Loads every point's ID, dense vector, and payload once. At 796 chunks x 384
    dims this is cheap (~1.2MB of floats) and lets every subsequent similarity
    search run as a single vectorized numpy operation against the FULL corpus,
    rather than a per-query Qdrant round-trip limited to an approximate top-K.
    """
    all_ids = []
    all_vectors = []
    payload_by_id = {}

    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION,
            limit=200,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        for p in points:
            all_ids.append(str(p.id))
            all_vectors.append(p.vector["dense"])
            payload_by_id[str(p.id)] = p.payload
        if offset is None:
            break

    return all_ids, np.array(all_vectors), payload_by_id


def find_distractor_for_band(
    all_chunk_ids: list[str],
    corpus_norms: np.ndarray,
    query_norm: np.ndarray,
    golden_chunk_id: str,
    target_similarity: float,
    already_used_ids: set[str],
) -> dict | None:
    similarities = corpus_norms @ query_norm

    excluded = already_used_ids | {golden_chunk_id}
    candidate_mask = np.array([cid not in excluded for cid in all_chunk_ids])
    if not candidate_mask.any():
        return None

    candidate_indices = np.where(candidate_mask)[0]
    candidate_sims = similarities[candidate_indices]
    best_local_idx = np.argmin(np.abs(candidate_sims - target_similarity))
    best_idx = candidate_indices[best_local_idx]
    best_sim = float(similarities[best_idx])

    if abs(best_sim - target_similarity) > BAND_TOLERANCE * 3:
        return None

    return {
        "chunk_id": all_chunk_ids[best_idx],
        "similarity_to_query": best_sim,
        "target_band_similarity": target_similarity,
    }


def build_raft_record(
    all_chunk_ids: list[str],
    corpus_norms: np.ndarray,
    payload_by_id: dict[str, dict],
    embedder: SentenceTransformer,
    record: dict,
    is_abstain: bool,
) -> dict | None:
    chunk_id = record["chunk_id"]
    query = record["candidate"]["instruction"]
    real_answer = record["candidate"]["output"]

    if chunk_id not in payload_by_id:
        return None
    golden_payload = payload_by_id[chunk_id]

    query_vector = np.array(embedder.encode(query))
    query_norm = query_vector / np.linalg.norm(query_vector)

    distractors = []
    used_ids = {chunk_id}
    for band_name, target_sim in BANDS.items():
        distractor = find_distractor_for_band(
            all_chunk_ids, corpus_norms, query_norm, chunk_id, target_sim, used_ids
        )
        if distractor is None:
            continue
        distractor["band"] = band_name
        distractor_payload = payload_by_id[distractor["chunk_id"]]
        distractor["text"] = distractor_payload.get("text", "")
        distractor["source"] = distractor_payload.get("source", "")
        distractors.append(distractor)
        used_ids.add(distractor["chunk_id"])

    raft_record = {
        "chunk_id": chunk_id,
        "query": query,
        "distractors": distractors,
        "bands_found": [d["band"] for d in distractors],
        "bands_missing": [b for b in BANDS if b not in [d["band"] for d in distractors]],
        "is_abstain": is_abstain,
    }

    if is_abstain:
        raft_record["golden_chunk"] = None
        raft_record["answer"] = ABSTAIN_ANSWER
    else:
        raft_record["golden_chunk"] = {
            "chunk_id": chunk_id,
            "text": golden_payload.get("text", ""),
            "source": golden_payload.get("source", ""),
        }
        raft_record["answer"] = real_answer

    return raft_record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--abstain-fraction", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--source", default="accepted_dataset.jsonl")
    parser.add_argument("--output", default="raft_records.jsonl")
    args = parser.parse_args()

    random.seed(SEED)

    source_path = Path(args.source)
    if not source_path.exists():
        print(f"FATAL: {source_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    records = load_records(source_path, args.limit)
    print(f"Loaded {len(records)} records from {source_path}")

    client = QdrantClient(
        url=os.environ.get("QDRANT_URL"), api_key=os.environ.get("QDRANT_API_KEY")
    )
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading full corpus (IDs, vectors, payloads) once...")
    all_chunk_ids, all_vectors, payload_by_id = load_full_corpus(client)
    print(f"Loaded {len(all_chunk_ids)} chunks from Qdrant")
    corpus_norms = all_vectors / np.linalg.norm(all_vectors, axis=1, keepdims=True)

    abstain_ids = set(
        random.sample([r["chunk_id"] for r in records], int(len(records) * args.abstain_fraction))
    )
    print(
        f"Selected {len(abstain_ids)} records ({args.abstain_fraction:.0%}) for abstain conversion"
    )

    output_path = Path(args.output)
    written = 0
    incomplete_bands = 0
    skipped = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for i, record in enumerate(records):
            if (i + 1) % 100 == 0:
                print(f"--- progress: {i + 1}/{len(records)} ---")

            is_abstain = record["chunk_id"] in abstain_ids
            raft_record = build_raft_record(
                all_chunk_ids, corpus_norms, payload_by_id, embedder, record, is_abstain
            )

            if raft_record is None:
                skipped += 1
                continue

            if raft_record["bands_missing"]:
                incomplete_bands += 1

            out_f.write(json.dumps(raft_record) + "\n")
            written += 1

    print("\n" + "=" * 60)
    print(f"Written: {written}")
    print(f"Skipped (golden chunk not found in Qdrant): {skipped}")
    print(f"Records with at least one missing band: {incomplete_bands}")
    print(f"Abstain records: {len(abstain_ids)}")
    print(f"Output: {output_path.resolve()}")


if __name__ == "__main__":
    main()
