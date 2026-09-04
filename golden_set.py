import json

from pipeline import ask_llm, hybrid_search_with_rerank

questions = [
    "How many breeds of dogs are there?",
    "What's the scientific name for the wolf?",
    "Why do cats sleep so much?",
    "What color can a wolf's fur be?",
    "How long has coffee been cultivated?",
    "What continent did coffee originate from?",
    "Do dogs have a better sense of smell than humans?",
    "What is the DSM-5 diagnosis related to caffeine?",
    "What's the relationship between dogs and wolves?",
    "How many teeth do dogs have?",
    "What does the parasite-mediated domestication hypothesis suggest?",
    "What's the difference between the 2023 and 2026 remote work policies?",
    "What is espresso?",
    "Why were dogs originally domesticated, according to the commensal pathway theory?",
    "What's the product code for the coffee maker, and how many units were made?",
    "Do cats have a social survival strategy like herd behavior?",
    "What happened during the French officer's coffee voyage in 1723?",
    "What percentage of caffeine users develop tolerance to its sleep effects?",
    "What is domestication syndrome?",
    "Can dogs communicate with humans?",
]

results = []
for q in questions:
    retrieved = hybrid_search_with_rerank(q, tenant_id=None)
    answer, _ = ask_llm(q, retrieved["documents"][0])
    results.append({"question": q, "answer": answer})
    print(f"Done: {q}")

with open("golden_set_answers.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved {len(results)} question/answer pairs to golden_set_answers.json")
