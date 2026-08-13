from dotenv import load_dotenv
import os
from qdrant_client import QdrantClient

load_dotenv(override=True)

qdrant_url = os.getenv("QDRANT_URL")
qdrant_key = os.getenv("QDRANT_API_KEY")

print(f"URL loaded: {qdrant_url is not None}")
print(f"Key loaded: {qdrant_key is not None}")

client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
collections = client.get_collections()
print(f"\nConnection successful. Existing collections: {collections}")