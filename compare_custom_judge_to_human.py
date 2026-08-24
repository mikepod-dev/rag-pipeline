import json

with open("human_grades.json", "r") as f:
    human = json.load(f)

with open("custom_judge_results.json", "r") as f:
    custom = json.load(f)

assert len(human) == len(custom), "Mismatched grade counts - something went wrong"

FAITHFULNESS_PASS_THRESHOLD = 0.5

agree = 0
disagreements = []

for h, c in zip(human, custom):
    assert h["question"] == c["question"], "Question order mismatch"

    custom_grade = "PASS" if c["faithfulness"] > FAITHFULNESS_PASS_THRESHOLD else "FAIL"

    if h["human_grade"] == custom_grade:
        agree += 1
    else:
        disagreements.append(
            {
                "question": h["question"],
                "human_grade": h["human_grade"],
                "custom_grade": custom_grade,
                "faithfulness_score": c["faithfulness"],
                "rationale": c["rationale"],
                "answer": h["answer"],
            }
        )

total = len(human)
raw_agreement = agree / total

human_pass_rate = sum(1 for h in human if h["human_grade"] == "PASS") / total
custom_pass_rate = sum(1 for c in custom if c["faithfulness"] > FAITHFULNESS_PASS_THRESHOLD) / total

p_agree_by_chance = (human_pass_rate * custom_pass_rate) + (
    (1 - human_pass_rate) * (1 - custom_pass_rate)
)
kappa = (
    (raw_agreement - p_agree_by_chance) / (1 - p_agree_by_chance) if p_agree_by_chance < 1 else 0
)

print(f"Total cases: {total}")
print(f"Faithfulness PASS threshold: > {FAITHFULNESS_PASS_THRESHOLD}")
print(f"Raw agreement (human vs. custom_judge): {agree}/{total} ({raw_agreement:.1%})")
print(f"Human PASS rate: {human_pass_rate:.1%}")
print(f"custom_judge PASS rate: {custom_pass_rate:.1%}")
print(f"Cohen's kappa (human vs. custom_judge): {kappa:.3f}")

print(f"\n--- Disagreements ({len(disagreements)}) ---")
for d in disagreements:
    print(f"\nQ: {d['question']}")
    print(
        f"  Human: {d['human_grade']} | custom_judge: {d['custom_grade']} (faithfulness={d['faithfulness_score']})"
    )
    print(f"  Rationale: {d['rationale']}")
