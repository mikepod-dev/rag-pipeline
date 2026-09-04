from pipeline import hybrid_search

result = hybrid_search(
    "What year did the French officer bring coffee to the Americas?", tenant_id=None
)
print("Documents returned:", len(result["documents"][0]))
for i, (doc, meta) in enumerate(zip(result["documents"][0], result["metadatas"][0])):
    print(f"\n--- Result {i+1} ({meta['source']}) ---")
    print(doc[:150])
