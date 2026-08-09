import os
import json
import time
from datetime import datetime
from rank_bm25 import BM25Okapi

def load_documents(folder):
    documents = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            documents.append({"source": filename, "text": content})
    return documents

docs = load_documents("docs")
print(f"Loaded {len(docs)} documents")

def chunk_text(text, chunk_size=100, overlap=30):
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

all_chunks = []
for doc in docs:
    doc_chunks = chunk_text(doc["text"])
    for chunk in doc_chunks:
        all_chunks.append({"source": doc["source"], "text": chunk})

print(f"Total chunks: {len(all_chunks)}")

from sentence_transformers import SentenceTransformer, CrossEncoder

model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

for chunk in all_chunks:
    embedding = model.encode(chunk["text"])
    chunk["embedding"] = embedding

print(f"Embedding length: {len(all_chunks[0]['embedding'])}")
print(all_chunks[0]["embedding"][:5])

tokenized_chunks = [chunk["text"].lower().split() for chunk in all_chunks]
bm25 = BM25Okapi(tokenized_chunks)

import chromadb

client = chromadb.Client()
collection = client.create_collection(name="my_docs")

for i, chunk in enumerate(all_chunks):
    collection.add(
        ids=[str(i)],
        embeddings=[chunk["embedding"].tolist()],
        documents=[chunk["text"]],
        metadatas=[{"source": chunk["source"]}]
    )

def hybrid_search(query, n_results=2, k=60):
    query_embedding = model.encode(query).tolist()
    vector_results = collection.query(query_embeddings=[query_embedding], n_results=len(all_chunks))
    vector_ranking = [int(doc_id) for doc_id in vector_results["ids"][0]]

    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_ranking = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)

    rrf_scores = {}
    for rank, idx in enumerate(vector_ranking):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rank + k)
    for rank, idx in enumerate(bm25_ranking):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rank + k)

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, score in ranked[:n_results]]

    return {
        "documents": [[all_chunks[i]["text"] for i in top_indices]],
        "metadatas": [[{"source": all_chunks[i]["source"]} for i in top_indices]]
    }

def hybrid_search_with_rerank(query, n_candidates=25, n_final=2):
    wide_results = hybrid_search(query, n_results=n_candidates)
    candidates = wide_results["documents"][0]
    candidate_sources = wide_results["metadatas"][0]

    pairs = [[query, doc] for doc in candidates]
    scores = reranker.predict(pairs)

    scored = list(zip(candidates, candidate_sources, scores))
    scored.sort(key=lambda x: x[2], reverse=True)

    top = scored[:n_final]
    return {
        "documents": [[doc for doc, src, score in top]],
        "metadatas": [[src for doc, src, score in top]]
    }

def compare_search(query, n_results=2):
    query_embedding = model.encode(query).tolist()
    semantic_only = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    hybrid = hybrid_search(query, n_results=n_results)

    print(f"\nQuery: {query}")
    print("Semantic-only top sources:", [m["source"] for m in semantic_only["metadatas"][0]])
    print("Hybrid top sources:", [m["source"] for m in hybrid["metadatas"][0]])

from dotenv import load_dotenv
import requests

load_dotenv(override=True)
api_key = os.getenv("OPENROUTER_API_KEY")

total_cost = 0.0
total_calls = 0
cache_hits = 0
answer_cache = {}

def ask_llm(question, context_chunks):
    global total_cost, total_calls, cache_hits

    cache_key = question.lower().strip()
    if cache_key in answer_cache:
        cache_hits += 1
        return answer_cache[cache_key]

    context = "\n\n".join(context_chunks)

    prompt = f"""Answer the question using ONLY the following context.

If the context contains conflicting or contradictory information, do not silently pick one - explicitly state that there is a conflict, and briefly describe what each source says.

Context:
{context}

Question: {question}"""

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "~anthropic/claude-haiku-latest",
            "messages": [{"role": "user", "content": prompt}]
        }
    )
    data = response.json()

    call_cost = data.get("usage", {}).get("cost", 0)
    total_cost += call_cost
    total_calls += 1

    answer_text = data["choices"][0]["message"]["content"]

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer_text,
        "cost": call_cost
    }
    with open("query_log.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    answer_cache[cache_key] = answer_text
    return answer_text

def validate_query(query):
    if not query or not query.strip():
        return False, "Question cannot be empty."
    if len(query) > 500:
        return False, "Question is too long (max 500 characters)."
    return True, None

if __name__ == "__main__":
    while True:
        query = input("\nAsk a question (or type 'quit'): ")
        if query.lower() == "quit":
            print(f"\nSession total: {total_calls} calls, ${total_cost:.6f}, {cache_hits} cache hits")
            break

        is_valid, error_message = validate_query(query)
        if not is_valid:
            print(f"\nError: {error_message}")
            continue

        retrieval_start = time.time()
        query_embedding = model.encode(query).tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=2)
        retrieved_texts = results["documents"][0]
        retrieval_time = time.time() - retrieval_start

        generation_start = time.time()
        answer = ask_llm(query, retrieved_texts)
        generation_time = time.time() - generation_start

        print("\nANSWER:", answer)
        print(f"\n(Retrieval: {retrieval_time:.2f}s | Generation: {generation_time:.2f}s)")