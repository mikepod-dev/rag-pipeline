"""
inject_tenant_canaries.py

Module 4: injects two real, embedded canary points -- one tagged tenant_a,
one tagged tenant_b -- each with a unique, made-up fact that cannot appear
anywhere else in the real corpus. This is the same canary discipline as
Finding 34's geometry-watchdog canary: a deliberately injected, known-answer
probe, not an unverified assumption that isolation works.

One-time setup for the cross-tenant leakage test. Run once before
test_tenant_isolation.py.
"""

import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Document, PointStruct
from sentence_transformers import SentenceTransformer

load_dotenv(override=True)

COLLECTION = "rag_docs_hybrid"

CANARIES = [
    {
        "id": "11111111-1111-1111-1111-111111111111",
        "tenant_id": "tenant_a",
        "source": "canary_tenant_a.txt",
        "text": "The secret onboarding code for Zephyrsoft Industries is QUARTZ-7742.",
    },
    {
        "id": "22222222-2222-2222-2222-222222222222",
        "tenant_id": "tenant_b",
        "source": "canary_tenant_b.txt",
        "text": "The secret onboarding code for Halcyon Ventures is FALCON-3391.",
    },
]


def main():
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])

    points = []
    for canary in CANARIES:
        embedding = model.encode(canary["text"]).tolist()
        points.append(
            PointStruct(
                id=canary["id"],
                vector={
                    "dense": embedding,
                    "sparse": Document(text=canary["text"], model="Qdrant/bm25"),
                },
                payload={
                    "text": canary["text"],
                    "source": canary["source"],
                    "tenant_id": canary["tenant_id"],
                },
            )
        )
        print(f"Prepared canary for {canary['tenant_id']}: {canary['text']!r}")

    client.upsert(collection_name=COLLECTION, points=points)
    print(f"\nUpserted {len(points)} canary points to '{COLLECTION}'.")


if __name__ == "__main__":
    main()
