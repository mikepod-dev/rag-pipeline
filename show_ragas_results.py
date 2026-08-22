import json

results = json.load(open("ragas_results.json"))
for r in results:
    print(f"{r['faithfulness']:.3f}  {r['answer_relevancy']:.3f}  {r['user_input']}")
