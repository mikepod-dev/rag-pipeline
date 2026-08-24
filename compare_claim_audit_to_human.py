import json

with open("human_grades.json", "r") as f:
    human = json.load(f)

with open("claim_audit_results.json", "r") as f:
    claim_audit = json.load(f)

assert len(human) == len(claim_audit), "Mismatched grade counts - something went wrong"

SCORE_PASS_THRESHOLD = 0.5

agree = 0
disagreements = []

for h, c in zip(human, claim_audit):
    assert h["question"] == c["question"], "Question order mismatch"

    grade = "PASS" if c["score"] > SCORE_PASS_THRESHOLD else "FAIL"

    if h["human_grade"] == grade:
        agree += 1
    else:
        disagreements.append(
            {
                "question": h["question"],
                "human_grade": h["human_grade"],
                "claim_audit_grade": grade,
                "score": c["score"],
                "grounded_claims": c["grounded_claims"],
                "total_claims": c["total_claims"],
                "answer": h["answer"],
            }
        )

total = len(human)
raw_agreement = agree / total

human_pass_rate = sum(1 for h in human if h["human_grade"] == "PASS") / total
claim_audit_pass_rate = sum(1 for c in claim_audit if c["score"] > SCORE_PASS_THRESHOLD) / total

p_agree_by_chance = (human_pass_rate * claim_audit_pass_rate) + (
    (1 - human_pass_rate) * (1 - claim_audit_pass_rate)
)
kappa = (
    (raw_agreement - p_agree_by_chance) / (1 - p_agree_by_chance) if p_agree_by_chance < 1 else 0
)

print(f"Total cases: {total}")
print(f"Score PASS threshold: > {SCORE_PASS_THRESHOLD}")
print(f"Raw agreement (human vs. claim_audit): {agree}/{total} ({raw_agreement:.1%})")
print(f"Human PASS rate: {human_pass_rate:.1%}")
print(f"claim_audit PASS rate: {claim_audit_pass_rate:.1%}")
print(f"Cohen's kappa (human vs. claim_audit): {kappa:.3f}")

print(f"\n--- Disagreements ({len(disagreements)}) ---")
for d in disagreements:
    print(f"\nQ: {d['question']}")
    print(
        f"  Human: {d['human_grade']} | claim_audit: {d['claim_audit_grade']} (score={d['score']:.3f}, {d['grounded_claims']}/{d['total_claims']} claims)"
    )
