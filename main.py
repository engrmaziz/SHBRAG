import os
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from src.config import COLLECTION_NAME, EMBEDDING_MODEL, GROQ_MODEL
from src.ingest import process_and_upload_pdf
from src.healing import autonomous_rag_pipeline

app = FastAPI(title="SHBRAG Autonomous Pipeline")

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
async def upload_document(file: UploadFile = File(...)):
    # Save the file temporarily
    os.makedirs("data", exist_ok=True)
    temp_filepath = f"data/{file.filename}"
    
    with open(temp_filepath, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
        
    # Process and upload to Qdrant Cloud
    chunks_processed = process_and_upload_pdf(temp_filepath)
    
    # Cleanup temporary file
    os.remove(temp_filepath)
    
    return {
        "status": "success",
        "message": f"Successfully ingested {file.filename}", 
        "chunks_embedded": chunks_processed
    }

@app.post("/api/v1/ask")
async def ask_question(request: QueryRequest):
    # Pass the query to the self-healing orchestrator
    result = autonomous_rag_pipeline(request.query)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)