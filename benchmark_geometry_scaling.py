"""
benchmark_geometry_scaling.py

Module 2 (Tier 4): actually measure real latency and RSS memory for the geometry
watchdog's O(n^2) similarity-matrix computation at synthetic corpus sizes larger
than the real 796-chunk corpus, rather than trusting Finding 34's extrapolation
further than it's been tested.

Generates synthetic random unit vectors matching the real embedder's dimensionality
(384, all-MiniLM-L6-v2) and replicates geometry_watchdog.py's exact computation:
normalize, then a single matmul to produce the full n x n cosine similarity matrix.

Run as a SEPARATE PROCESS per corpus size (not a loop within one process) so RSS
measurements at each size start from a clean baseline and don't compound.

Usage:
    python benchmark_geometry_scaling.py --n 5000
    python benchmark_geometry_scaling.py --n 5000 --append results.jsonl
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import psutil

EMBEDDING_DIM = 384  # matches all-MiniLM-L6-v2, this project's real embedder


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1e6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True, help="synthetic corpus size")
    parser.add_argument("--append", type=str, default=None, help="JSONL file to append result to")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rss_start = rss_mb()
    print(f"n={args.n:,} | RSS at start: {rss_start:.1f} MB")

    rng = np.random.default_rng(args.seed)
    # float64 to match geometry_watchdog.py's real behavior: np.array() on a list of
    # Python floats (what Qdrant's client returns) defaults to float64, not float32.
    vectors = rng.standard_normal((args.n, EMBEDDING_DIM))
    rss_after_gen = rss_mb()
    print(
        f"RSS after generating {args.n:,} synthetic vectors: {rss_after_gen:.1f} MB "
        f"(+{rss_after_gen - rss_start:.1f} MB)"
    )

    t0 = time.time()
    norms = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    sim_matrix = norms @ norms.T
    elapsed = time.time() - t0
    rss_after_matmul = rss_mb()

    print(f"Computed {args.n:,}x{args.n:,} similarity matrix in {elapsed:.4f}s")
    print(
        f"RSS after matmul: {rss_after_matmul:.1f} MB "
        f"(+{rss_after_matmul - rss_after_gen:.1f} MB for the matrix itself)"
    )

    theoretical_matrix_mb = (args.n**2 * 8) / 1e6
    print(f"Theoretical matrix size (n^2 * 8 bytes, float64): {theoretical_matrix_mb:.1f} MB")
    print(
        f"Real vs theoretical ratio: {(rss_after_matmul - rss_after_gen) / theoretical_matrix_mb:.2f}x"
    )

    result = {
        "n": args.n,
        "rss_start_mb": round(rss_start, 1),
        "rss_after_gen_mb": round(rss_after_gen, 1),
        "rss_after_matmul_mb": round(rss_after_matmul, 1),
        "matmul_seconds": round(elapsed, 4),
        "theoretical_matrix_mb": round(theoretical_matrix_mb, 1),
    }

    if args.append:
        path = Path(args.append)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
        print(f"\nAppended result to {path.resolve()}")

    del sim_matrix, norms, vectors
    print(f"RSS after explicit del: {rss_mb():.1f} MB")


if __name__ == "__main__":
    main()
