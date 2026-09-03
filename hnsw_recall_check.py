"""
hnsw_recall_check.py

Module 2 (Tier 4), second question: does Qdrant's default approximate HNSW search
recover the same near-duplicate pairs the brute-force geometry watchdog already
found real evidence for (Finding 34), and what does recall@K look like generally
across the real corpus -- measured against Qdrant's own exact (server-side
brute-force) search as ground truth, not assumed.

Two real checks:
  1. Known-pair recall: for every pair already flagged by geometry_watchdog.py
     (loaded from clumping_report.json), query Qdrant's approximate search from
     both member points and check whether the other member appears in the
     top-K results. This is the operationally relevant question -- would HNSW
     have caught what brute force already caught, on the real corpus.
  2. General recall@K: for every real chunk (or a random sample), compare
     approximate vs. exact (SearchParams(exact=True)) top-K neighbor sets and
     score deviation, aggregated across the corpus.

Usage:
    python hnsw_recall_check.py --k 20
    python hnsw_recall_check.py --k 20 --sample 200
"""

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv(override=True)

COLLECTION = "rag_docs_hybrid"


def load_full_corpus(client: QdrantClient):
    all_ids, all_vectors = [], []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION,
            limit=200,
            offset=offset,
            with_payload=False,
            with_vectors=True,
        )
        for p in points:
            all_ids.append(str(p.id))
            all_vectors.append(p.vector["dense"])
        if offset is None:
            break
    return all_ids, all_vectors


def query_neighbors(client, vector, k, exact):
    result = client.query_points(
        collection_name=COLLECTION,
        query=vector,
        using="dense",
        limit=k + 1,  # +1 since the point's own vector will match itself, score 1.0
        search_params=models.SearchParams(exact=exact) if exact else None,
    )
    return result.points


def check_known_pairs(client, k):
    report_path = Path("clumping_report.json")
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    pairs = report["flagged_pairs"]
    print(f"\n=== Known-pair recall check (k={k}) ===")
    print(f"Checking {len(pairs)} pairs already flagged by geometry_watchdog.py (Finding 34)")

    caught_either_direction = 0
    caught_both_directions = 0
    for pair in pairs:
        id1, id2 = pair["chunk_id_1"], pair["chunk_id_2"]

        # get each chunk's stored vector via a point-id lookup (retrieve), not a fresh query
        retrieved = client.retrieve(collection_name=COLLECTION, ids=[id1, id2], with_vectors=True)
        vec_by_id = {str(p.id): p.vector["dense"] for p in retrieved}
        if id1 not in vec_by_id or id2 not in vec_by_id:
            print(f"  WARNING: could not retrieve vectors for pair ({id1}, {id2}) -- skipping")
            continue

        neighbors_from_1 = {
            str(pt.id) for pt in query_neighbors(client, vec_by_id[id1], k, exact=False)
        }
        neighbors_from_2 = {
            str(pt.id) for pt in query_neighbors(client, vec_by_id[id2], k, exact=False)
        }

        found_from_1 = id2 in neighbors_from_1
        found_from_2 = id1 in neighbors_from_2

        if found_from_1 or found_from_2:
            caught_either_direction += 1
        if found_from_1 and found_from_2:
            caught_both_directions += 1

        status = (
            "BOTH"
            if (found_from_1 and found_from_2)
            else ("ONE-WAY" if (found_from_1 or found_from_2) else "MISSED")
        )
        print(f"  sim={pair['similarity']:.4f}  {status}  ({id1[:8]}... <-> {id2[:8]}...)")

    print(f"\nCaught in at least one direction: {caught_either_direction}/{len(pairs)}")
    print(f"Caught in both directions:        {caught_both_directions}/{len(pairs)}")
    return {
        "k": k,
        "total_known_pairs": len(pairs),
        "caught_either_direction": caught_either_direction,
        "caught_both_directions": caught_both_directions,
    }


def general_recall_at_k(client, all_ids, all_vectors, k, sample):
    print(f"\n=== General recall@{k} check ===")
    n = len(all_ids)
    if sample and sample < n:
        import random

        random.seed(42)
        sample_idx = random.sample(range(n), sample)
        print(f"Sampling {sample} of {n} real chunks")
    else:
        sample_idx = range(n)
        print(f"Checking all {n} real chunks")

    recalls = []
    score_deviations = []

    t0 = time.time()
    for i in sample_idx:
        chunk_id, vector = all_ids[i], all_vectors[i]

        approx = query_neighbors(client, vector, k, exact=False)
        exact = query_neighbors(client, vector, k, exact=True)

        approx_by_id = {str(p.id): p.score for p in approx if str(p.id) != chunk_id}
        exact_by_id = {str(p.id): p.score for p in exact if str(p.id) != chunk_id}

        approx_ids = set(list(approx_by_id.keys())[:k])
        exact_ids = set(list(exact_by_id.keys())[:k])

        overlap = approx_ids & exact_ids
        recall = len(overlap) / k if k > 0 else 0.0
        recalls.append(recall)

        for oid in overlap:
            score_deviations.append(abs(approx_by_id[oid] - exact_by_id[oid]))

    elapsed = time.time() - t0

    import statistics

    mean_recall = statistics.mean(recalls) if recalls else 0.0
    min_recall = min(recalls) if recalls else 0.0
    mean_score_dev = statistics.mean(score_deviations) if score_deviations else 0.0
    max_score_dev = max(score_deviations) if score_deviations else 0.0

    print(f"Queries checked: {len(recalls)} (each = 1 approx + 1 exact query)")
    print(
        f"Total time: {elapsed:.2f}s ({elapsed / max(len(recalls), 1):.4f}s per chunk pair of queries)"
    )
    print(f"Mean recall@{k}: {mean_recall:.4f}")
    print(f"Min recall@{k}:  {min_recall:.4f}")
    print(f"Mean score deviation (matched neighbors): {mean_score_dev:.6f}")
    print(f"Max score deviation (matched neighbors):  {max_score_dev:.6f}")

    return {
        "k": k,
        "n_checked": len(recalls),
        "elapsed_seconds": elapsed,
        "mean_recall": mean_recall,
        "min_recall": min_recall,
        "mean_score_deviation": mean_score_dev,
        "max_score_deviation": max_score_dev,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument(
        "--sample", type=int, default=None, help="sample size; default = all chunks"
    )
    args = parser.parse_args()

    client = QdrantClient(
        url=os.environ.get("QDRANT_URL"), api_key=os.environ.get("QDRANT_API_KEY")
    )

    known_pair_result = check_known_pairs(client, args.k)

    print("\nLoading full corpus for general recall check...")
    all_ids, all_vectors = load_full_corpus(client)
    print(f"Loaded {len(all_ids)} real chunks")

    general_result = general_recall_at_k(client, all_ids, all_vectors, args.k, args.sample)

    report = {
        "known_pair_check": known_pair_result,
        "general_recall_check": general_result,
    }
    with open("hnsw_recall_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("\nReport written: hnsw_recall_report.json")


if __name__ == "__main__":
    main()
