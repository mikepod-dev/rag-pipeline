"""
geometry_watchdog_explore.py

Module 4, phase 1: compute the FULL 796x796 pairwise cosine similarity matrix
against the real Qdrant corpus and look at the actual distribution before picking
a clumping threshold -- the curriculum's ~0.92 is explicitly described as
illustrative ("tune this empirically against your embedder, don't cargo-cult the
number"), and this project has already been burned once this session by assuming
a plausible-sounding default number applies to this specific corpus without
checking (the RAFT similarity bands).

Categorizes flagged pairs by same-source vs cross-source, since same-document
adjacent-chunk similarity is expected (chunking overlap by design) and is a
different phenomenon from genuine cross-topic confusion (the "malaria vs flu"
case the curriculum is actually worried about).
"""

import os
import time

import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv(override=True)

COLLECTION = "rag_docs_hybrid"


def load_full_corpus(client: QdrantClient) -> tuple[list[str], np.ndarray, list[str]]:
    all_ids = []
    all_vectors = []
    all_sources = []

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
        if offset is None:
            break

    return all_ids, np.array(all_vectors), all_sources


def main():
    client = QdrantClient(
        url=os.environ.get("QDRANT_URL"), api_key=os.environ.get("QDRANT_API_KEY")
    )

    print("Loading full corpus...")
    all_ids, all_vectors, all_sources = load_full_corpus(client)
    n = len(all_ids)
    print(f"Loaded {n} chunks, vector dim {all_vectors.shape[1]}")

    print("\nComputing full pairwise cosine similarity matrix...")
    t0 = time.time()
    norms = all_vectors / np.linalg.norm(all_vectors, axis=1, keepdims=True)
    sim_matrix = norms @ norms.T
    elapsed = time.time() - t0
    n_pairs = n * (n - 1) // 2
    print(f"Computed {n_pairs} unique pairs in {elapsed:.3f}s ({n}x{n} matrix)")

    # Upper triangle only, excluding self-pairs (diagonal is always 1.0)
    iu = np.triu_indices(n, k=1)
    pair_sims = sim_matrix[iu]

    print("\n=== Full distribution across all pairs ===")
    print(f"Min:    {pair_sims.min():.4f}")
    print(f"p50:    {np.percentile(pair_sims, 50):.4f}")
    print(f"p90:    {np.percentile(pair_sims, 90):.4f}")
    print(f"p99:    {np.percentile(pair_sims, 99):.4f}")
    print(f"p99.9:  {np.percentile(pair_sims, 99.9):.4f}")
    print(f"Max:    {pair_sims.max():.4f}")
    print(f"Mean:   {pair_sims.mean():.4f}")

    print("\n=== Candidate threshold counts (all pairs, same+cross source combined) ===")
    for threshold in [0.80, 0.85, 0.90, 0.92, 0.95, 0.98]:
        count = (pair_sims > threshold).sum()
        print(f"  > {threshold}: {count} pairs ({100 * count / n_pairs:.3f}%)")

    print("\n=== Same-source vs cross-source breakdown at each threshold ===")
    row_idx, col_idx = iu
    same_source_mask = np.array(
        [all_sources[row_idx[i]] == all_sources[col_idx[i]] for i in range(len(row_idx))]
    )

    for threshold in [0.80, 0.85, 0.90, 0.92, 0.95, 0.98]:
        above = pair_sims > threshold
        same = (above & same_source_mask).sum()
        cross = (above & ~same_source_mask).sum()
        print(f"  > {threshold}: {same} same-source, {cross} cross-source")

    print(
        "\n=== Top 10 highest-similarity CROSS-source pairs (the real 'malaria vs flu' candidates) ==="
    )
    cross_only_sims = np.where(same_source_mask, -1, pair_sims)
    top_cross_idx = np.argsort(cross_only_sims)[::-1][:10]
    for idx in top_cross_idx:
        r, c = row_idx[idx], col_idx[idx]
        print(
            f"  sim={pair_sims[idx]:.4f}  {all_sources[r]} <-> {all_sources[c]}  "
            f"[{all_ids[r][:8]}... <-> {all_ids[c][:8]}...]"
        )

    print(f"\nCompute time for full {n}x{n} matrix: {elapsed:.4f}s")

    print("\n" + "=" * 70)
    print("=== FULL TEXT of the single highest-similarity cross-source pair ===")
    top_idx = top_cross_idx[0]
    r, c = row_idx[top_idx], col_idx[top_idx]

    points_by_id = {}
    offset = None
    while True:
        pts, offset = client.scroll(
            collection_name=COLLECTION,
            limit=200,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in pts:
            points_by_id[str(p.id)] = p.payload.get("text", "")
        if offset is None:
            break

    print(f"\n[{all_sources[r]}] (id={all_ids[r]}):")
    print(points_by_id.get(all_ids[r], "TEXT NOT FOUND"))
    print(f"\n[{all_sources[c]}] (id={all_ids[c]}):")
    print(points_by_id.get(all_ids[c], "TEXT NOT FOUND"))

    print("\n" + "=" * 70)
    print("=== Top 5 highest-similarity SAME-source pairs (chunking-overlap bug candidates) ===")
    same_only_sims = np.where(same_source_mask, pair_sims, -1)
    top_same_idx = np.argsort(same_only_sims)[::-1][:5]
    for idx in top_same_idx:
        r, c = row_idx[idx], col_idx[idx]
        print(
            f"  sim={pair_sims[idx]:.4f}  {all_sources[r]}  [{all_ids[r][:8]}... <-> {all_ids[c][:8]}...]"
        )

    print("\n" + "=" * 70)
    print("=== FULL TEXT of the single highest-similarity same-source pair ===")
    top_same = top_same_idx[0]
    r, c = row_idx[top_same], col_idx[top_same]
    print(f"\n[{all_sources[r]}] chunk 1 (id={all_ids[r]}):")
    print(points_by_id.get(all_ids[r], "TEXT NOT FOUND"))
    print(f"\n[{all_sources[c]}] chunk 2 (id={all_ids[c]}):")
    print(points_by_id.get(all_ids[c], "TEXT NOT FOUND"))


if __name__ == "__main__":
    main()
