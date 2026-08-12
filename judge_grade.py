import json
import requests
from pipeline import api_key

with open("golden_set_answers.json", "r") as f:
    data = json.load(f)

def judge_answer_standalone(question, answer):
    prompt = f"""You are grading an AI assistant's answer for factual correctness only.

Question: {question}
Answer given: {answer}

Does this answer correctly and substantively address the question? Reply with ONLY one word: PASS or FAIL."""

    votes = []
    for _ in range(3):
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "~anthropic/claude-haiku-latest",
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        data_response = response.json()
        verdict = data_response["choices"][0]["message"]["content"]
        votes.append(verdict.strip().upper())

    pass_count = sum(1 for v in votes if "PASS" in v)
    return "PASS" if pass_count >= 2 else "FAIL", votes

judge_grades = []
print("Running 3-way judge on all 20 questions...\n")

for i, item in enumerate(data):
    verdict, votes = judge_answer_standalone(item["question"], item["answer"])
    judge_grades.append({
        "question": item["question"],
        "answer": item["answer"],
        "judge_grade": verdict,
        "votes": votes
    })
    print(f"{i+1}/20: {item['question'][:50]}... -> {verdict} (votes: {votes})")

with open("judge_grades.json", "w") as f:
    json.dump(judge_grades, f, indent=2)

print("\nSaved judge grades to judge_grades.json")