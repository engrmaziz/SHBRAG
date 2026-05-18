from fastapi import FastAPI

from src.config import COLLECTION_NAME, EMBEDDING_MODEL, GROQ_MODEL

app = FastAPI(title="SHBRAG")


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "collection": COLLECTION_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "groq_model": GROQ_MODEL,
    }
