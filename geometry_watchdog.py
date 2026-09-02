"""
geometry_watchdog.py

Module 4: pairwise embedding-similarity watchdog. Computes the full 796x796 cosine
similarity matrix against the real corpus, flags pairs above a threshold empirically
chosen for THIS corpus/embedder (0.80 -- the curriculum's illustrative 0.92 only
catches 2 pairs total here, too sparse for meaningful analysis; see geometry_watchdog_explore.py
for the real distribution that justified this choice), and separates same-source
(expected: chunking overlap) from cross-source (the real "malaria vs flu" concern) pairs.

Includes a canary test: injects one deliberately duplicated chunk (verbatim copy of a
real chunk's text and vector, given a different synthetic source label) before running
the real corpus through the watchdog. If the canary isn't flagged, the detection
mechanism itself is broken and any "zero true positives" result on the real corpus
would be meaningless -- this is checked BEFORE trusting the real corpus's results,
per the curriculum's own explicit warning about this exact failure mode.

Exits non-zero if any cross-source pair exceeds the threshold, matching the
"pre-retrieval CI check" framing -- this could gate a real CI/CD pipeline on new
chunk ingestion. Writes a machine-readable clumping_report.json regardless of outcome.

Usage:
    python geometry_watchdog.py --threshold 0.80
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv(override=True)

COLLECTION = "rag_docs_hybrid"
CANARY_SOURCE_LABEL = "__canary_injected__"


def load_full_corpus(
    client: QdrantClient,
) -> tuple[list[str], np.ndarray, list[str], dict[str, str]]:
    all_ids, all_vectors, all_sources = [], [], []
    text_by_id = {}

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
            all_sources.append(p.payload.get("source", "UNKNOWN"))
            text_by_id[str(p.id)] = p.payload.get("text", "")
        if offset is None:
            break

    return all_ids, np.array(all_vectors), all_sources, text_by_id


def inject_canary(
    all_ids: list[str], all_vectors: np.ndarray, all_sources: list[str], text_by_id: dict[str, str]
) -> tuple[list[str], np.ndarray, list[str], str]:
    """
    Duplicates chunk 0's vector and text verbatim under a synthetic source label and a
    fake ID. A verbatim duplicate MUST score ~1.0 similarity against its source -- if
    this isn't flagged, nothing else the watchdog reports can be trusted.
    """
    canary_id = "canary-injected-0000"
    canary_vector = all_vectors[0].copy()
    text_by_id[canary_id] = text_by_id[all_ids[0]]

    new_ids = all_ids + [canary_id]
    new_vectors = np.vstack([all_vectors, canary_vector[None, :]])
    new_sources = all_sources + [CANARY_SOURCE_LABEL]

    return new_ids, new_vectors, new_sources, canary_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--false-positive-sample-size", type=int, default=50)
    args = parser.parse_args()

    client = QdrantClient(
        url=os.environ.get("QDRANT_URL"), api_key=os.environ.get("QDRANT_API_KEY")
    )

    print("Loading full corpus...")
    all_ids, all_vectors, all_sources, text_by_id = load_full_corpus(client)
    n_real = len(all_ids)
    print(f"Loaded {n_real} real chunks")

    print("Injecting canary (verbatim duplicate of chunk 0, synthetic source label)...")
    all_ids, all_vectors, all_sources, canary_id = inject_canary(
        all_ids, all_vectors, all_sources, text_by_id
    )
    n = len(all_ids)

    print(f"\nComputing full {n}x{n} pairwise similarity matrix (including canary)...")
    t0 = time.time()
    norms = all_vectors / np.linalg.norm(all_vectors, axis=1, keepdims=True)
    sim_matrix = norms @ norms.T
    elapsed = time.time() - t0
    print(f"Computed in {elapsed:.4f}s")

    iu = np.triu_indices(n, k=1)
    row_idx, col_idx = iu
    pair_sims = sim_matrix[iu]
    same_source_mask = np.array(
        [all_sources[row_idx[i]] == all_sources[col_idx[i]] for i in range(len(row_idx))]
    )

    # --- Canary check: does the watchdog actually catch a known, deliberate duplicate? ---
    canary_idx_in_ids = all_ids.index(canary_id)
    canary_pair_mask = (row_idx == canary_idx_in_ids) | (col_idx == canary_idx_in_ids)
    canary_sims = pair_sims[canary_pair_mask]
    canary_max_sim = canary_sims.max() if len(canary_sims) else 0.0
    canary_caught = canary_max_sim > args.threshold

    print("\n=== CANARY CHECK ===")
    print(f"Canary's max similarity to any real chunk: {canary_max_sim:.4f}")
    print(f"Canary caught at threshold {args.threshold}: {canary_caught}")
    if not canary_caught:
        print(
            "FATAL: canary was NOT caught. The detection mechanism itself is broken.",
            file=sys.stderr,
        )
        print(
            "A 'zero true positives' result on the real corpus would be meaningless.",
            file=sys.stderr,
        )
        sys.exit(2)

    # --- Real corpus flagging (excluding all canary-involved pairs from real analysis) ---
    real_mask = ~canary_pair_mask
    real_pair_sims = pair_sims[real_mask]
    real_same_source = same_source_mask[real_mask]
    real_row = row_idx[real_mask]
    real_col = col_idx[real_mask]

    above_threshold = real_pair_sims > args.threshold
    cross_source_flags = above_threshold & ~real_same_source
    same_source_flags = above_threshold & real_same_source

    print(f"\n=== REAL CORPUS RESULTS (threshold={args.threshold}) ===")
    print(f"Same-source flagged (expected: chunking overlap): {same_source_flags.sum()}")
    print(f"Cross-source flagged (the real clumping concern): {cross_source_flags.sum()}")

    flagged_pairs = []
    for i in np.where(above_threshold)[0]:
        r, c = real_row[i], real_col[i]
        flagged_pairs.append(
            {
                "chunk_id_1": all_ids[r],
                "chunk_id_2": all_ids[c],
                "source_1": all_sources[r],
                "source_2": all_sources[c],
                "similarity": float(real_pair_sims[i]),
                "same_source": bool(real_same_source[i]),
            }
        )
    flagged_pairs.sort(key=lambda x: x["similarity"], reverse=True)

    report = {
        "threshold": args.threshold,
        "n_real_chunks": n_real,
        "canary_max_similarity": float(canary_max_sim),
        "canary_caught": bool(canary_caught),
        "total_flagged": len(flagged_pairs),
        "same_source_flagged": int(same_source_flags.sum()),
        "cross_source_flagged": int(cross_source_flags.sum()),
        "compute_time_seconds": elapsed,
        "flagged_pairs": flagged_pairs,
    }

    report_path = Path("clumping_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written: {report_path.resolve()}")

    # O(n^2) scaling wall, extrapolated from the real measured compute time at this corpus size
    time_per_pair = elapsed / (n * (n - 1) / 2)
    print("\n=== O(n^2) scaling wall (extrapolated from real measured compute time) ===")
    for target_n in [1_000, 5_000, 10_000, 50_000, 100_000]:
        target_pairs = target_n * (target_n - 1) / 2
        target_seconds = target_pairs * time_per_pair
        target_memory_mb = (target_n**2 * 8) / 1e6  # float64 similarity matrix
        print(
            f"  n={target_n:,}: ~{target_seconds:.2f}s compute, ~{target_memory_mb:.1f} MB for the full matrix"
        )

    cross_source_real_flags = int(cross_source_flags.sum())
    if cross_source_real_flags > 0:
        print(
            f"\nEXIT NON-ZERO: {cross_source_real_flags} cross-source pair(s) exceeded threshold."
        )
        sys.exit(1)
    else:
        print("\nNo cross-source clumping detected above threshold. Exiting 0.")
        sys.exit(0)


if __name__ == "__main__":
    main()
