import hashlib
import json
import os
import threading
import time
import uuid
from datetime import datetime

import requests
from dotenv import load_dotenv

from celery_app import celery_app

load_dotenv(override=True)

api_key = os.getenv("OPENROUTER_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_key = os.getenv("QDRANT_API_KEY")

collection_name = "rag_docs_hybrid"

# These stay None until initialize() actually runs - importing this module
# alone (e.g. to use chunk_text or validate_query) triggers no network calls.
docs = None
all_chunks = None
model = None
reranker = None
client = None
bm25 = None
_initialized = False
_init_lock = threading.Lock()
_reranker_lock = threading.Lock()

MANIFEST_PATH = "ingestion_manifest.json"
# Fixed namespace so uuid5(source, chunk_index) is stable across every run
POINT_NAMESPACE = uuid.UUID("6f5905f2-3b1f-4a3e-9c1a-1f6b2a9d6a10")


def compute_file_hash(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha256.update(block)
    return sha256.hexdigest()


def load_manifest():
    if not os.path.exists(MANIFEST_PATH):
        return {}
    with open(MANIFEST_PATH, "r") as f:
        return json.load(f)


def save_manifest(manifest):
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def chunk_point_id(source, chunk_index):
    return str(uuid.uuid5(POINT_NAMESPACE, f"{source}::{chunk_index}"))


def load_documents(folder):
    documents = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            documents.append({"source": filename, "text": content})
    return documents


def chunk_text(text, chunk_size=100, overlap=30):
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks


def validate_query(query):
    if not query or not query.strip():
        return False, "Question cannot be empty."
    if len(query) > 500:
        return False, "Question is too long (max 500 characters)."
    return True, None


def initialize():
    """Loads documents, computes embeddings, and connects to Qdrant.
    Only runs once, and only when something actually needs it -
    not on plain `import pipeline`."""
    global docs, all_chunks, model, reranker, client, _initialized

    if _initialized:
        return

    with _init_lock:
        if _initialized:
            return

        from qdrant_client import QdrantClient, models
        from qdrant_client.models import (
            Distance,
            Document,
            PointStruct,
            SparseVectorParams,
            VectorParams,
        )
        from sentence_transformers import SentenceTransformer

        folder = "docs"
        manifest = load_manifest()
        current_files = os.listdir(folder)

        changed_sources = []
        unchanged_sources = []
        for filename in current_files:
            filepath = os.path.join(folder, filename)
            file_hash = compute_file_hash(filepath)
            if manifest.get(filename, {}).get("hash") != file_hash:
                changed_sources.append(filename)
            else:
                unchanged_sources.append(filename)

        removed_sources = [f for f in manifest if f not in current_files]

        print(
            f"Delta check: {len(changed_sources)} changed, "
            f"{len(unchanged_sources)} unchanged, {len(removed_sources)} removed"
        )

        docs = []
        for filename in changed_sources:
            filepath = os.path.join(folder, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            docs.append({"source": filename, "text": content})

        all_chunks = []
        for doc in docs:
            for i, chunk in enumerate(chunk_text(doc["text"])):
                all_chunks.append({"source": doc["source"], "text": chunk, "chunk_index": i})
        print(
            f"Re-chunked {len(all_chunks)} chunks from {len(changed_sources)} changed document(s)"
        )

        model = SentenceTransformer("all-MiniLM-L6-v2")

        for chunk in all_chunks:
            embedding = model.encode(chunk["text"])
            chunk["embedding"] = embedding
        if all_chunks:
            print(f"Embedding length: {len(all_chunks[0]['embedding'])}")

        client = QdrantClient(url=qdrant_url, api_key=qdrant_key)

        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config={"dense": VectorParams(size=384, distance=Distance.COSINE)},
                sparse_vectors_config={"sparse": SparseVectorParams()},
            )

        client.create_payload_index(
            collection_name=collection_name,
            field_name="source",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        sources_to_clear = changed_sources + removed_sources
        if sources_to_clear:
            client.delete(
                collection_name=collection_name,
                points_selector=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source", match=models.MatchAny(any=sources_to_clear)
                        )
                    ]
                ),
            )
            print(f"Cleared existing points for {len(sources_to_clear)} changed/removed source(s)")

        if all_chunks:
            points = [
                PointStruct(
                    id=chunk_point_id(chunk["source"], chunk["chunk_index"]),
                    vector={
                        "dense": chunk["embedding"].tolist(),
                        "sparse": Document(text=chunk["text"], model="Qdrant/bm25"),
                    },
                    payload={"text": chunk["text"], "source": chunk["source"]},
                )
                for chunk in all_chunks
            ]
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                client.upsert(collection_name=collection_name, points=batch)
                print(f"Upserted batch {i // batch_size + 1} ({len(batch)} points)")
            print(f"Upserted {len(points)} total points to Qdrant collection '{collection_name}'")
        else:
            print("No changed documents - nothing to upsert.")

        now = datetime.now().isoformat()
        for filename in changed_sources:
            manifest[filename] = {
                "hash": compute_file_hash(os.path.join(folder, filename)),
                "last_processed": now,
            }
        for filename in removed_sources:
            del manifest[filename]
        save_manifest(manifest)

        # stash the models module reference for query-time use (Prefetch, FusionQuery, etc.)
        globals()["_qmodels"] = models

        _initialized = True


def get_reranker():
    """Lazily loads the cross-encoder reranker on first use, not at initialize() time.
    Thread-safe via double-checked locking, same pattern as initialize()."""
    global reranker

    if reranker is not None:
        return reranker

    with _reranker_lock:
        if reranker is not None:
            return reranker

        from sentence_transformers import CrossEncoder

        print("Lazy-loading reranker on first use...")
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return reranker


def hybrid_search(query, n_results=2, max_per_source=3, prefetch_limit=50):
    initialize()
    models = globals()["_qmodels"]

    query_embedding = model.encode(query).tolist()

    results = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(query=query_embedding, using="dense", limit=prefetch_limit),
            models.Prefetch(
                query=models.Document(text=query, model="Qdrant/bm25"),
                using="sparse",
                limit=prefetch_limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=prefetch_limit,
    )

    source_counts = {}
    documents = []
    metadatas = []
    for point in results.points:
        source = point.payload["source"]
        if source_counts.get(source, 0) >= max_per_source:
            continue
        documents.append(point.payload["text"])
        metadatas.append({"source": source})
        source_counts[source] = source_counts.get(source, 0) + 1
        if len(documents) >= n_results:
            break

    return {"documents": [documents], "metadatas": [metadatas]}


def hybrid_search_with_rerank(query, n_candidates=25, n_final=2, max_per_source=3):
    wide_results = hybrid_search(query, n_results=n_candidates, max_per_source=max_per_source)
    candidates = wide_results["documents"][0]
    candidate_sources = wide_results["metadatas"][0]

    pairs = [[query, doc] for doc in candidates]
    scores = get_reranker().predict(pairs)

    scored = list(zip(candidates, candidate_sources, scores))
    scored.sort(key=lambda x: x[2], reverse=True)

    top = scored[:n_final]
    return {
        "documents": [[doc for doc, src, score in top]],
        "metadatas": [[src for doc, src, score in top]],
    }


def compare_search(query, n_results=2):
    initialize()
    query_embedding = model.encode(query).tolist()
    semantic_only = client.query_points(
        collection_name=collection_name, query=query_embedding, using="dense", limit=n_results
    )

    print(f"\nQuery: {query}")
    print("Semantic-only top sources:", [point.payload["source"] for point in semantic_only.points])


total_cost = 0.0
total_calls = 0
cache_hits = 0
answer_cache = {}


def ask_llm(question, context_chunks):
    global total_cost, total_calls, cache_hits

    cache_key = question.lower().strip()
    if cache_key in answer_cache:
        cache_hits += 1
        return answer_cache[cache_key], 0.0

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
            "messages": [{"role": "user", "content": prompt}],
        },
    )
    data = response.json()

    call_cost = data.get("usage", {}).get("cost", 0)
    total_cost += call_cost
    total_calls += 1

    answer_text = data["choices"][0]["message"]["content"]

    answer_cache[cache_key] = answer_text
    return answer_text, call_cost


def log_query_metrics(question, answer, cost, latency_ms, success, retrieval_count):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "cost": cost,
        "latency_ms": latency_ms,
        "success": success,
        "retrieval_count": retrieval_count,
    }
    with open("query_log.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")


def self_grade_answer(question, answer, retrieved_context):
    context_preview = "\n\n".join(retrieved_context)
    grade_prompt = f"""You are evaluating whether a RAG system's answer is as good as it could be, given what was actually retrieved.

Question: {question}
Retrieved context: {context_preview}
Answer given: {answer}

If the answer says information is missing, check: does the retrieved context actually look unrelated to the question (in which case the refusal is CORRECT and this is SUFFICIENT), or does the context seem like it's on-topic but the answer still failed to use it well (in which case retrieval likely grabbed the wrong specific chunks, and this is INSUFFICIENT - a different search might do better)?

Reply with ONLY one word: SUFFICIENT or INSUFFICIENT."""

    grade, _ = ask_llm(grade_prompt, [])
    return "SUFFICIENT" in grade.strip().upper()


def rewrite_query(original_question, failed_answer):
    rewrite_prompt = f"""The following question was asked, but the answer given was insufficient - likely because retrieval didn't find the right information.

Original question: {original_question}
Insufficient answer: {failed_answer}

Rewrite the question using different words, broader terms, or a more direct phrasing that might retrieve better source material. Reply with ONLY the rewritten question, nothing else."""

    rewritten, _ = ask_llm(rewrite_prompt, [])
    return rewritten.strip()


def agentic_answer(question, max_retries=2):
    current_question = question
    attempts = []

    for attempt in range(max_retries + 1):
        results = hybrid_search_with_rerank(current_question)
        retrieved_texts = results["documents"][0]
        answer, _ = ask_llm(current_question, retrieved_texts)

        is_sufficient = self_grade_answer(question, answer, retrieved_texts)
        attempts.append(
            {
                "attempt": attempt + 1,
                "question_used": current_question,
                "answer": answer,
                "sufficient": is_sufficient,
            }
        )

        if is_sufficient:
            return answer, attempts

        if attempt < max_retries:
            current_question = rewrite_query(current_question, answer)

    return answer, attempts


@celery_app.task(name="pipeline.answer_question_task")
def answer_question_task(question):
    start = time.time()
    success = True
    answer = None
    cost = 0.0
    retrieval_count = 0

    try:
        results = hybrid_search_with_rerank(question)
        retrieved_texts = results["documents"][0]
        retrieval_count = len(retrieved_texts)
        answer, cost = ask_llm(question, retrieved_texts)
    except Exception:
        success = False
        raise
    finally:
        latency_ms = (time.time() - start) * 1000
        log_query_metrics(question, answer, cost, latency_ms, success, retrieval_count)

    return {"question": question, "answer": answer}


if __name__ == "__main__":
    initialize()
    while True:
        query = input("\nAsk a question (or type 'quit'): ")
        if query.lower() == "quit":
            print(
                f"\nSession total: {total_calls} calls, ${total_cost:.6f}, {cache_hits} cache hits"
            )
            break

        is_valid, error_message = validate_query(query)
        if not is_valid:
            print(f"\nError: {error_message}")
            continue

        overall_start = time.time()
        success = True
        answer = None
        cost = 0.0
        retrieval_count = 0

        try:
            retrieval_start = time.time()
            results = hybrid_search_with_rerank(query)
            retrieved_texts = results["documents"][0]
            retrieval_count = len(retrieved_texts)
            retrieval_time = time.time() - retrieval_start

            generation_start = time.time()
            answer, cost = ask_llm(query, retrieved_texts)
            generation_time = time.time() - generation_start

            print("\nANSWER:", answer)
            print(f"\n(Retrieval: {retrieval_time:.2f}s | Generation: {generation_time:.2f}s)")
        except Exception as e:
            success = False
            print(f"\nError: {e}")
        finally:
            latency_ms = (time.time() - overall_start) * 1000
            log_query_metrics(query, answer, cost, latency_ms, success, retrieval_count)
