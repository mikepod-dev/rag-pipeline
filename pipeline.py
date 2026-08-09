import os
from rank_bm25 import BM25Okapi

def load_documents(folder):
    documents = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        with open(filepath, "r") as f:
            content = f.read()
            documents.append({"source": filename, "text": content})
    return documents

docs = load_documents("docs")
print(docs)

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
for c in all_chunks:
    print(c)

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

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

def hybrid_search(query, n_results=2):
    query_embedding = model.encode(query).tolist()
    vector_results = collection.query(query_embeddings=[query_embedding], n_results=len(all_chunks))

    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    combined_scores = {}
    for i, doc_id in enumerate(vector_results["ids"][0]):
        idx = int(doc_id)
        vector_distance = vector_results["distances"][0][i]
        vector_score = 1 / (1 + vector_distance)
        bm25_score = bm25_scores[idx]
        combined_scores[idx] = vector_score + (bm25_score * 0.1)

    ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, score in ranked[:n_results]]

    return {
        "documents": [[all_chunks[i]["text"] for i in top_indices]],
        "metadatas": [[{"source": all_chunks[i]["source"]} for i in top_indices]]
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

def ask_llm(question, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"Answer the question using ONLY the following context.\n\nContext:\n{context}\n\nQuestion: {question}"

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "~anthropic/claude-haiku-latest",
            "messages": [{"role": "user", "content": prompt}]
        }
    )
    data = response.json()
    return data["choices"][0]["message"]["content"]

if __name__ == "__main__":
    while True:
        query = input("\nAsk a question (or type 'quit'): ")
        if query.lower() == "quit":
            break

        query_embedding = model.encode(query).tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=2)
        retrieved_texts = results["documents"][0]

        answer = ask_llm(query, retrieved_texts)
        print("\nANSWER:", answer)