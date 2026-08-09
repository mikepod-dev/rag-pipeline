from pipeline import model, collection, ask_llm, hybrid_search, hybrid_search_with_rerank, compare_search

eval_set = [
    {
        "question": "How long have dogs been domesticated?",
        "expected_source": None,
        "expected_answer": "Should give a real timeframe, roughly 14,000-25,000+ years, citing archaeological or genetic evidence.",
        "must_contain_any": [["14,000"], ["25,000"]]
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
        "expected_answer": "Should say this cannot be answered from the provided context.",
        "must_contain": ["cannot", "context"]
    },
    {
        "question": "Canine origins?",
        "expected_source": None,
        "expected_answer": "Should say dogs were bred from wolves.",
        "must_contain": ["wolves"]
    },
    {
        "question": "Does caffeine affect sleep?",
        "expected_source": None,
        "expected_answer": "Should say yes, caffeine affects/disrupts sleep, based on real evidence now in the context.",
        "must_contain_any": [["disrupt"], ["delay"], ["affect"]]
    },
    {
        "question": "Are wild dogs domesticated?",
        "expected_source": "animals_overview.txt",
        "expected_answer": "Should say wild dogs are NOT domesticated.",
        "must_contain": ["not domesticated"]
    },
    {
        "question": "Do cats hunt in packs?",
        "expected_source": "animals_overview.txt",
        "expected_answer": "Should say cats hunt alone/solitary, not in packs.",
        "must_contain": ["solitary"]
    },
    {
        "question": "What makes an animal a mammal?",
        "expected_source": "animals_overview.txt",
        "expected_answer": "Should mention warm-blooded and milk.",
        "must_contain": ["warm-blooded", "milk"]
    },
    {
        "question": "What year did the French officer bring coffee to the Americas, and what dangers did the voyage include?",
        "expected_source": "history_long.txt",
        "expected_answer": "Should say 1723, and mention a storm, a suspected saboteur, and a drought.",
        "must_contain": ["1723", "storm", "saboteur", "drought"]
    },
    {
        "question": "What is the product code for the coffee maker?",
        "expected_source": "facts.txt",
        "expected_answer": "Should say XJ-4471.",
        "must_contain": ["XJ-4471"]
    },
    {
        "question": "How many days per week can employees work remotely?",
        "expected_source": None,
        "expected_answer": "Should explicitly flag that there is a conflict between the 2023 (2 days) and 2026 (4 days) policies, not silently pick one.",
        "must_contain_any": [["conflict"], ["contradict"], ["two policy"], ["2023"]]
    },
]
def judge_answer(question, expected_answer, actual_answer):
    judge_prompt = f"""You are grading an AI's answer for factual correctness only. Ignore formatting or style.

Question: {question}
Expected answer should contain: {expected_answer}
Actual answer given: {actual_answer}

Does the actual answer's CONTENT match what's expected? Reply with ONLY one word: PASS or FAIL."""

    votes = []
    for _ in range(3):
        verdict = ask_llm(judge_prompt, [])
        votes.append(verdict.strip().upper())

    pass_count = sum(1 for v in votes if "PASS" in v)
    return "PASS" if pass_count >= 2 else "FAIL"
results_log = []

for case in eval_set:
    results = hybrid_search_with_rerank(case["question"])

    top_source = results["metadatas"][0][0]["source"]
    retrieved_texts = results["documents"][0]
    all_retrieved_sources = [m["source"] for m in results["metadatas"][0]]
    retrieval_correct = (case["expected_source"] is None) or (case["expected_source"] in all_retrieved_sources)

    answer = ask_llm(case["question"], retrieved_texts)

    if "must_contain" in case:
        answer_lower = answer.lower()
        all_present = all(term.lower() in answer_lower for term in case["must_contain"])
        verdict = "PASS" if all_present else "FAIL"
    elif "must_contain_any" in case:
        answer_lower = answer.lower()
        any_phrase_matched = any(
            all(term.lower() in answer_lower for term in phrase_group)
            for phrase_group in case["must_contain_any"]
        )
        verdict = "PASS" if any_phrase_matched else "FAIL"
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
compare_search("What makes an animal a mammal?", n_results=25)