"""Healing utilities for SHBRAG."""

from __future__ import annotations

from src.retrieve import generate_answer, retrieve_context


def evaluate_confidence(retrieved_chunks: list[dict]) -> bool:
    """Evaluate retrieval quality using average vector similarity score."""
    if not retrieved_chunks:
        return False

    average_score = sum(float(chunk.get("score", 0.0)) for chunk in retrieved_chunks) / len(
        retrieved_chunks
    )
    return average_score >= 0.65


def autonomous_rag_pipeline(user_query: str) -> dict:
    """Run a self-healing RAG flow with retrieval and generation fallbacks."""
    chunks = retrieve_context(user_query, top_k=3)

    if not evaluate_confidence(chunks):
        rewritten_query = f"{user_query} (rewrite for better semantic search)"
        chunks = retrieve_context(rewritten_query, top_k=5)

    for chunk in chunks:
        if "source" not in chunk:
            chunk["source"] = chunk.get("source_file", "")

    generated_answer = generate_answer(user_query, chunks)
    if generated_answer.strip() == "INSUFFICIENT_CONTEXT":
        return {
            "status": "failed",
            "reason": "Hallucination prevented. No relevant data in vector space.",
            "original_query": user_query,
        }

    return {
        "status": "success",
        "answer": generated_answer,
        "sources_used": [chunk["source"] for chunk in chunks],
    }
