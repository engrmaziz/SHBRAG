"""Retrieval utilities for SHBRAG."""

from __future__ import annotations

import httpx
from groq import Groq
from qdrant_client import QdrantClient

from src.config import (
    COLLECTION_NAME,
    GROQ_API_KEY,
    GROQ_MODEL,
    QDRANT_API_KEY,
    QDRANT_URL,
)
from src.ingest import get_hf_embedding

if not QDRANT_URL:
    raise ValueError("QDRANT_URL is not set.")

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY, http_client=httpx.Client())


def retrieve_context(query: str, top_k: int = 3) -> list[dict]:
    """Retrieve the most relevant text chunks from Qdrant Cloud."""
    if top_k <= 0:
        return []

    query_embedding = get_hf_embedding(query)
    results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k,
        with_payload=True,
    )

    context_chunks: list[dict] = []
    for result in results:
        payload = result.payload or {}
        context_chunks.append(
            {
                "text": payload.get("text", ""),
                "score": float(result.score),
            }
        )
    return context_chunks


def generate_answer(query: str, context_chunks: list[dict]) -> str:
    """Generate a grounded answer from retrieved context chunks via Groq."""
    system_prompt = (
        "Act as an expert research assistant. "
        "Answer the user's question using ONLY the provided context. "
        "If the context does not contain the answer, explicitly state: "
        "'INSUFFICIENT_CONTEXT'."
    )
    context_text = "\n\n".join(
        f"[Chunk {index + 1}] {chunk.get('text', '')}" for index, chunk in enumerate(context_chunks)
    )

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion:\n{query}",
            },
        ],
    )
    return response.choices[0].message.content or ""
