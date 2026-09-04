from eval import eval_set
from pipeline import hybrid_search_with_rerank


def run_retrieval_sweep(cap_values=[1, 2, 3, 4, 5, 999]):
    cases_with_source = [c for c in eval_set if c["expected_source"] is not None]
    print(
        f"Sweeping diversity cap over {len(cases_with_source)} eval cases with a known expected source\n"
    )

    results = {}
    for cap in cap_values:
        correct = 0
        for case in cases_with_source:
            retrieved = hybrid_search_with_rerank(
                case["question"], tenant_id=None, max_per_source=cap
            )
            sources = [m["source"] for m in retrieved["metadatas"][0]]
            if case["expected_source"] in sources:
                correct += 1
        accuracy = correct / len(cases_with_source)
        results[cap] = accuracy
        print(
            f"max_per_source={cap}: retrieval accuracy = {correct}/{len(cases_with_source)} ({accuracy:.1%})"
        )

    best_cap = max(results, key=results.get)
    print(f"\nOptimal cap: {best_cap} (accuracy: {results[best_cap]:.1%})")
    return results


if __name__ == "__main__":
    run_retrieval_sweep()
