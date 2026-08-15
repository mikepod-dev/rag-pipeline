import json

with open("golden_set_answers.json", "r") as f:
    data = json.load(f)

human_grades = []

print("=== BLIND GRADING ===")
print(
    "For each question, read the answer and judge for yourself: does it actually, correctly answer the question?"
)
print("Type 'p' for PASS, 'f' for FAIL, based on your own judgment.\n")

for i, item in enumerate(data):
    print(f"\n--- Question {i+1}/20 ---")
    print(f"Q: {item['question']}")
    print(f"A: {item['answer']}")
    grade = input("\nYour grade (p=PASS, f=FAIL): ").strip().lower()
    human_grades.append(
        {
            "question": item["question"],
            "answer": item["answer"],
            "human_grade": "PASS" if grade == "p" else "FAIL",
        }
    )

with open("human_grades.json", "w") as f:
    json.dump(human_grades, f, indent=2)

print("\nSaved your 20 grades to human_grades.json")
