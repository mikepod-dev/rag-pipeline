from pipeline import agentic_answer

question = "What's the weird theory about bugs inside animals causing them to become pets?"
answer, attempts = agentic_answer(question)

for a in attempts:
    print(f"\nAttempt {a['attempt']}: question_used='{a['question_used']}'")
    print(f"  Sufficient: {a['sufficient']}")
    print(f"  Answer: {a['answer'][:300]}")

print(f"\nFINAL ANSWER: {answer}")
