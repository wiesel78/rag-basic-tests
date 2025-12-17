import os
import uuid
import hashlib
from pathlib import Path
from typing import List, Dict, Any

import requests
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-large")
COLLECTION = os.getenv("QDRANT_COLLECTION", "docs")
DOCS_DIR = os.getenv("DOCS_DIR", "./docs")

# Chunking controls (tune these)
CHUNK_SIZE_CHARS = int(os.getenv("CHUNK_SIZE_CHARS", "1200"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "200"))


def read_text_files(root: Path) -> List[Path]:
    exts = {".txt", ".md", ".text"}
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return sorted(files)


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = text.replace("\r\n", "\n")
    if chunk_size <= 0:
        return [text]
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 5)

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def ollama_embed(texts: List[str]) -> List[List[float]]:
    """
    Uses Ollama embeddings endpoint.
    Ollama API supports:
      POST /api/embeddings  {"model": "...", "prompt": "..."}
    We do it one-by-one for maximum compatibility.
    """
    vectors = []
    for t in texts:
        resp = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": OLLAMA_EMBED_MODEL, "prompt": t},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        vec = data.get("embedding")
        if not vec:
            raise RuntimeError(f"No embedding returned: {data}")
        vectors.append(vec)
    return vectors


def stable_point_id(file_path: str, chunk_index: int) -> str:
    """
    Stable ID so re-ingestion overwrites the same chunks (idempotent-ish).
    Qdrant point IDs can be UUID strings.
    """
    h = hashlib.sha256(f"{file_path}::{chunk_index}".encode("utf-8")).hexdigest()
    return str(uuid.UUID(h[:32]))  # first 128 bits -> UUID


def ensure_collection(client: QdrantClient, vector_size: int) -> None:
    existing = client.get_collections().collections
    if any(c.name == COLLECTION for c in existing):
        # Optional: validate vector size; if mismatch, you should recreate the collection.
        return

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def main():
    docs_root = Path(DOCS_DIR).resolve()
    files = read_text_files(docs_root)
    if not files:
        raise SystemExit(f"No .txt/.md files found under: {docs_root}")

    qdrant = QdrantClient(url=QDRANT_URL)

    # Probe embedding dimension from a sample call
    sample_vec = ollama_embed(["dimension probe"])[0]
    ensure_collection(qdrant, vector_size=len(sample_vec))

    batch_points: List[PointStruct] = []
    BATCH_SIZE = 64

    for fp in tqdm(files, desc="Files"):
        rel = str(fp.relative_to(docs_root))
        text = fp.read_text(encoding="utf-8", errors="replace")

        chunks = chunk_text(text, CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS)
        if not chunks:
            continue

        vectors = ollama_embed(chunks)

        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            pid = stable_point_id(rel, i)
            payload: Dict[str, Any] = {
                "source_file": rel,
                "chunk_index": i,
                "text": chunk,
            }
            batch_points.append(PointStruct(id=pid, vector=vec, payload=payload))

            if len(batch_points) >= BATCH_SIZE:
                qdrant.upsert(collection_name=COLLECTION, points=batch_points)
                batch_points.clear()

    if batch_points:
        qdrant.upsert(collection_name=COLLECTION, points=batch_points)

    info = qdrant.get_collection(COLLECTION)
    print(f"Done. Collection '{COLLECTION}' now has:")
    print(f" - points_count: {info.points_count}")
    print(f" - vectors size: {info.config.params.vectors.size}")


if __name__ == "__main__":
    main()
