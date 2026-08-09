from pipeline import model, collection, ask_llm, hybrid_search

eval_set = [
    {
        "question": "How long have dogs been domesticated?",
        "expected_source": "dog.txt",
        "expected_answer": "Should say the context doesn't give a specific timeframe, only that dogs were the first animals domesticated."
    },
    {
        "question": "How many hours do cats sleep?",
        "expected_source": "cat.txt",
        "expected_answer": "Should say 12 to 16 hours a day.",
        "must_contain": ["12", "16"]
    },
    {
        "question": "What's the capital of France?",
        "expected_source": None,
        "expected_answer": "Should say this cannot be answered from the provided context."
    },
    {
        "question": "Canine origins?",
        "expected_source": "dog.txt",
        "expected_answer": "Should say dogs were bred from wolves.",
        "must_contain": ["wolves"]
    },
    {
        "question": "Does caffeine affect sleep?",
        "expected_source": None,
        "expected_answer": "Should say this cannot be answered from the provided context, since no connection between caffeine and sleep is stated."
    },
    {
        "question": "Are wild dogs domesticated?",
        "expected_source": "animals_overview.txt",
        "expected_answer": "Should say wild dogs like the African wild dog are NOT domesticated and are endangered."
    },
    {
        "question": "Do cats hunt in packs?",
        "expected_source": "animals_overview.txt",
        "expected_answer": "Should say cats retain solitary hunting instincts, unlike dogs which are pack animals."
    },
    {
        "question": "What makes an animal a mammal?",
        "expected_source": "animals_overview.txt",
        "expected_answer": "Should mention warm-blooded, hair or fur, and producing milk for young."
    },
    {
        "question": "What year did the French officer bring coffee to the Americas, and what dangers did the voyage include?",
        "expected_source": "history_long.txt",
        "expected_answer": "Should say 1723, and mention a storm, a suspected saboteur, and a drought.",
        "must_contain": ["1723", "storm", "saboteur", "drought"]
    },
]
def judge_answer(question, expected_answer, actual_answer):
    judge_prompt = f"""You are grading an AI's answer.

Question: {question}
Expected answer should contain: {expected_answer}
Actual answer given: {actual_answer}

Does the actual answer's CONTENT match what's expected, regardless of formatting? Reply with ONLY one word: PASS or FAIL."""

    verdict = ask_llm(judge_prompt, [])
    return verdict.strip()

results_log = []

for case in eval_set:
    results = hybrid_search(case["question"], n_results=2)

    top_source = results["metadatas"][0][0]["source"]
    retrieved_texts = results["documents"][0]
    all_retrieved_sources = [m["source"] for m in results["metadatas"][0]]
    retrieval_correct = (case["expected_source"] is None) or (case["expected_source"] in all_retrieved_sources)

    answer = ask_llm(case["question"], retrieved_texts)

    if "must_contain" in case:
        answer_lower = answer.lower()
        all_present = all(term.lower() in answer_lower for term in case["must_contain"])
        verdict = "PASS" if all_present else "FAIL"
    else:
        verdict = judge_answer(case["question"], case["expected_answer"], answer)

    results_log.append({
        "question": case["question"],
        "retrieval_correct": retrieval_correct,
        "top_source": top_source,
        "expected_source": case["expected_source"],
        "verdict": verdict,
        "answer": answer
    })

print("\n--- EVAL RESULTS ---")
for r in results_log:
    print(f"\nQ: {r['question']}")
    if not r['retrieval_correct']:
        print(f"Retrieval: FAIL (got '{r['top_source']}', expected '{r['expected_source']}')")
    else:
        print("Retrieval: PASS")
    print(f"Answer Judge: {r['verdict']}")
    print(f"Answer: {r['answer']}")

total = len(results_log)
retrieval_passes = sum(1 for r in results_log if r["retrieval_correct"])
answer_passes = sum(1 for r in results_log if "PASS" in r["verdict"])

print(f"\n--- SUMMARY ---")
print(f"Retrieval accuracy: {retrieval_passes}/{total}")
print(f"Answer accuracy: {answer_passes}/{total}")