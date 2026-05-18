import os
import shutil
from pathlib import Path
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from src.config import COLLECTION_NAME, EMBEDDING_MODEL, GROQ_MODEL
from src.healing import autonomous_rag_pipeline
from src.ingest import process_and_upload_pdf

app = FastAPI(title="SHBRAG")


class QueryRequest(BaseModel):
    query: str


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "collection": COLLECTION_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "groq_model": GROQ_MODEL,
    }


@app.post("/api/v1/upload")
def upload_pdf(file: UploadFile = File(...)) -> dict[str, int | str]:
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "").suffix
    temp_path = data_dir / f"{uuid4().hex}{suffix}"

    try:
        with temp_path.open("wb") as output_buffer:
            shutil.copyfileobj(file.file, output_buffer)

        chunk_count = process_and_upload_pdf(str(temp_path))
        return {
            "status": "success",
            "message": "File processed and uploaded successfully.",
            "chunk_count": chunk_count,
        }
    finally:
        if temp_path.exists():
            os.remove(temp_path)


@app.post("/api/v1/ask")
def ask_question(request: QueryRequest) -> dict:
    return autonomous_rag_pipeline(request.query)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
