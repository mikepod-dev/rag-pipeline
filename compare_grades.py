import json

with open("human_grades.json", "r") as f:
    human = json.load(f)

with open("judge_grades.json", "r") as f:
    judge = json.load(f)

assert len(human) == len(judge), "Mismatched grade counts - something went wrong"

agree = 0
disagreements = []

for h, j in zip(human, judge):
    assert h["question"] == j["question"], "Question order mismatch"
    if h["human_grade"] == j["judge_grade"]:
        agree += 1
    else:
        disagreements.append({
            "question": h["question"],
            "human_grade": h["human_grade"],
            "judge_grade": j["judge_grade"],
            "answer": h["answer"]
        })

total = len(human)
raw_agreement = agree / total

human_pass_rate = sum(1 for h in human if h["human_grade"] == "PASS") / total
judge_pass_rate = sum(1 for j in judge if j["judge_grade"] == "PASS") / total

p_agree_by_chance = (human_pass_rate * judge_pass_rate) + ((1 - human_pass_rate) * (1 - judge_pass_rate))
kappa = (raw_agreement - p_agree_by_chance) / (1 - p_agree_by_chance) if p_agree_by_chance < 1 else 0

print(f"Total cases: {total}")
print(f"Raw agreement: {agree}/{total} ({raw_agreement:.1%})")
print(f"Human PASS rate: {human_pass_rate:.1%}")
print(f"Judge PASS rate: {judge_pass_rate:.1%}")
print(f"Cohen's kappa: {kappa:.3f}")

print(f"\n--- Disagreements ({len(disagreements)}) ---")
for d in disagreements:
    print(f"\nQ: {d['question']}")
    print(f"  Human: {d['human_grade']} | Judge: {d['judge_grade']}")
    print(f"  Answer: {d['answer'][:200]}")