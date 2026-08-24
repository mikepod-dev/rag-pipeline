import json
import time

from custom_judge import evaluate_query

with open("ragas_dataset.json", "r") as f:
    data = json.load(f)

results = []

for i, item in enumerate(data):
    result = evaluate_query(item["question"], item["answer"], item["contexts"])
    results.append(
        {
            "question": item["question"],
            "faithfulness": result["faithfulness"],
            "context_precision": result["context_precision"],
            "rationale": result["rationale"],
        }
    )
    print(f"Done {i + 1}/{len(data)}: {item['question']} -> faithfulness={result['faithfulness']}")
    time.sleep(0.5)  # light rate-limit courtesy, not a hard requirement

with open("custom_judge_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved {len(results)} results to custom_judge_results.json")
