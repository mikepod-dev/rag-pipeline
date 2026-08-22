import json

from pipeline import hybrid_search_with_rerank

with open("human_grades.json", "r") as f:
    original_data = json.load(f)

ragas_dataset = []

for i, item in enumerate(original_data):
    question = item["question"]
    answer = item["answer"]

    retrieved = hybrid_search_with_rerank(question)
    contexts = retrieved["documents"][0]

    ragas_dataset.append(
        {
            "question": question,
            "answer": answer,
            "contexts": contexts,
        }
    )
    print(f"Done {i + 1}/{len(original_data)}: {question}")

with open("ragas_dataset.json", "w") as f:
    json.dump(ragas_dataset, f, indent=2)

print(f"\nSaved {len(ragas_dataset)} entries to ragas_dataset.json")
