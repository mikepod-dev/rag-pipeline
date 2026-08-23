import json

with open("human_grades.json", "r") as f:
    human = json.load(f)

with open("ragas_results.json", "r") as f:
    ragas = json.load(f)

assert len(human) == len(ragas), "Mismatched grade counts - something went wrong"

# Reasoned, disclosed threshold: a faithfulness score of 0.5 or higher means
# a majority of the answer's claims were checked as grounded in the retrieved
# context. This is a defensible starting threshold, not a statistically
# derived one, in the same spirit as monitor.py's thresholds - it would be
# tuned against more data in a real deployment.
FAITHFULNESS_PASS_THRESHOLD = 0.5

agree = 0
disagreements = []

for h, r in zip(human, ragas):
    assert h["question"] == r["user_input"], "Question order mismatch"

    ragas_grade = "PASS" if r["faithfulness"] >= FAITHFULNESS_PASS_THRESHOLD else "FAIL"

    if h["human_grade"] == ragas_grade:
        agree += 1
    else:
        disagreements.append(
            {
                "question": h["question"],
                "human_grade": h["human_grade"],
                "ragas_grade": ragas_grade,
                "faithfulness_score": r["faithfulness"],
                "answer": h["answer"],
            }
        )

total = len(human)
raw_agreement = agree / total

human_pass_rate = sum(1 for h in human if h["human_grade"] == "PASS") / total
ragas_pass_rate = sum(1 for r in ragas if r["faithfulness"] >= FAITHFULNESS_PASS_THRESHOLD) / total

p_agree_by_chance = (human_pass_rate * ragas_pass_rate) + (
    (1 - human_pass_rate) * (1 - ragas_pass_rate)
)
kappa = (
    (raw_agreement - p_agree_by_chance) / (1 - p_agree_by_chance) if p_agree_by_chance < 1 else 0
)

print(f"Total cases: {total}")
print(f"Faithfulness PASS threshold: >= {FAITHFULNESS_PASS_THRESHOLD}")
print(f"Raw agreement (human vs. RAGAS faithfulness): {agree}/{total} ({raw_agreement:.1%})")
print(f"Human PASS rate: {human_pass_rate:.1%}")
print(f"RAGAS (faithfulness) PASS rate: {ragas_pass_rate:.1%}")
print(f"Cohen's kappa (human vs. RAGAS faithfulness): {kappa:.3f}")

print(f"\n--- Disagreements ({len(disagreements)}) ---")
for d in disagreements:
    print(f"\nQ: {d['question']}")
    print(
        f"  Human: {d['human_grade']} | RAGAS: {d['ragas_grade']} (faithfulness={d['faithfulness_score']:.3f})"
    )
    print(f"  Answer: {d['answer'][:200]}")
