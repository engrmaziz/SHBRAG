"""Ingestion utilities for SHBRAG."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pdfplumber
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from src.config import (
    COLLECTION_NAME,
    QDRANT_API_KEY,
    QDRANT_URL,
)

# Load the ultra-lightweight model locally (Runs on any basic CPU!)
print("Loading local embedding model...")
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Embedding model loaded successfully!")

def get_hf_embedding(text: str) -> list[float]:
    """Generate embeddings locally to completely bypass Hugging Face API errors."""
    if not text:
        raise ValueError("Text for embedding cannot be empty.")
    
    # Instantly converts text to a 384-dimension vector on your local CPU
    return embedding_model.encode(text).tolist()

def init_qdrant() -> QdrantClient:
    """Initialize Qdrant client and ensure collection exists."""
    if not QDRANT_URL:
        raise ValueError("QDRANT_URL is not set.")

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    collections = client.get_collections().collections
    collection_names = {collection.name for collection in collections}

    if COLLECTION_NAME not in collection_names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

    return client

def process_and_upload_pdf(filepath: str) -> int:
    """Extract, chunk, embed, and upload PDF content to Qdrant Cloud."""
    with pdfplumber.open(filepath) as pdf:
        full_text = "\n".join((page.extract_text() or "").strip() for page in pdf.pages).strip()

    if not full_text:
        return 0

    chunk_size = 500
    overlap = 50
    step = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(full_text), step):
        chunk = full_text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)

    if not chunks:
        return 0

    client = init_qdrant()
    source_file = Path(filepath).name
    points: list[PointStruct] = []
    
    for chunk in chunks:
        embedding = get_hf_embedding(chunk)
        points.append(
            PointStruct(
                id=str(uuid4()),
                vector=embedding,
                payload={"text": chunk, "source_file": source_file},
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
    return len(points)