import json
import time

from claim_audit_judge import evaluate_response

with open("ragas_dataset.json", "r") as f:
    data = json.load(f)

results = []

for i, item in enumerate(data):
    result = evaluate_response(item["question"], item["contexts"], item["answer"])
    results.append(
        {
            "question": item["question"],
            "score": result["score"],
            "total_claims": result["total_claims"],
            "grounded_claims": result["grounded_claims"],
            "claims_matrix": result["claims_matrix"],
            "internal_rationale": result["internal_rationale"],
        }
    )
    print(
        f"Done {i + 1}/{len(data)}: {item['question']} -> "
        f"score={result['score']:.3f} ({result['grounded_claims']}/{result['total_claims']} claims)"
    )
    time.sleep(0.5)

with open("claim_audit_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved {len(results)} results to claim_audit_results.json")
