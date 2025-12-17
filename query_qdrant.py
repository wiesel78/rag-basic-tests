import os
import requests
from qdrant_client import QdrantClient

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-large")
COLLECTION = os.getenv("QDRANT_COLLECTION", "docs")

def embed(q: str):
    r = requests.post(f"{OLLAMA_URL}/api/embeddings", json={"model": MODEL, "prompt": q}, timeout=120)
    r.raise_for_status()
    return r.json()["embedding"]

client = QdrantClient(url=QDRANT_URL)

query = "ask your question here"
vec = embed(query)

hits = client.search(collection_name=COLLECTION, query_vector=vec, limit=5)
for h in hits:
    print("score:", h.score, "file:", h.payload.get("source_file"), "chunk:", h.payload.get("chunk_index"))
    print(h.payload.get("text")[:400])
    print("---")
