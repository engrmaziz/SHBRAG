from pathlib import Path
import os

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

COLLECTION_NAME = "autonomous_rag"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
GROQ_MODEL = "llama-3.1-8b-instant"

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

missing_api_keys = [
    key_name
    for key_name, key_value in {
        "QDRANT_API_KEY": QDRANT_API_KEY,
        "GROQ_API_KEY": GROQ_API_KEY,
        "HF_API_KEY": HF_API_KEY,
    }.items()
    if not key_value
]

if missing_api_keys:
    raise ValueError(
        "Missing required API key(s): " + ", ".join(missing_api_keys)
    )
