"""Ingestion utilities for SHBRAG."""

from __future__ import annotations

import time
from pathlib import Path
from uuid import uuid4

import pdfplumber
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    HF_API_KEY,
    QDRANT_API_KEY,
    QDRANT_URL,
)


def get_hf_embedding(text: str) -> list[float]:
    """Get embedding vector from the Hugging Face Inference API with retries."""
    if not text:
        raise ValueError("Text for embedding cannot be empty.")

    url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBEDDING_MODEL}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": text}

    retries = 0
    max_retries = 3
    while True:
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            if response.status_code in {429, 503} and retries < max_retries:
                sleep_seconds = 2**retries
                time.sleep(sleep_seconds)
                retries += 1
                continue
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list) and data and isinstance(data[0], list):
                return [float(value) for value in data[0]]
            if isinstance(data, list):
                return [float(value) for value in data]
            raise ValueError("Unexpected embedding response format from Hugging Face API.")
        except requests.RequestException:
            if retries >= max_retries:
                raise
            sleep_seconds = 2**retries
            time.sleep(sleep_seconds)
            retries += 1


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
