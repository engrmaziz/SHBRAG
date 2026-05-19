## Title & Badges

# SHBRAG (Self-Healing Bootstrapped Retrieval-Augmented Generation)

[![Python Version](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Asynchronous%20Gateway-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Qdrant Cloud](https://img.shields.io/badge/Qdrant-Cloud%20Vector%20DB-DC244C?logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Groq Inference](https://img.shields.io/badge/Groq-Llama%203.1%20Inference-F55036)](https://console.groq.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Build Status](https://img.shields.io/badge/Hugging%20Face%20Spaces-Docker%20Build-blue?logo=huggingface)](https://huggingface.co/spaces)

## Executive Architectural Overview

SHBRAG is an autonomous, privacy-first RAG backend that blends **serverless API orchestration**, **local embedding generation**, and **cloud-native vector retrieval** into a deterministic inference pipeline. Its primary design objective is reducing hallucination risk before generation by enforcing a **mathematical confidence-threshold firewall** over retrieved semantic context.

At runtime, the service ingests unmanaged document assets, chunks and embeds them using a local CPU-safe transformer model, persists vectors in Qdrant Cloud, and validates retrieval quality before LLM inference through Groq-hosted Llama 3.1. If retrieval confidence is weak, SHBRAG automatically retries with a query-rewrite pass. If context is still insufficient, the system returns a deterministic failure frame instead of fabricating output.

> **Architectural definition:** SHBRAG favors grounded refusal over speculative generation when retrieval confidence or context sufficiency is below policy.

> **Operational constraint:** Required API keys (`QDRANT_API_KEY`, `GROQ_API_KEY`, `HF_API_KEY`) are validated at startup to prevent partial, unsafe execution states.

## Core System Architecture Diagram

```text
[Edge Trigger: Folder Watch/Webhook via Make.com]
                     |
                     v
      [Document Pull (Google Drive file fetch)]
                     |
                     v
     [POST /api/v1/upload -> FastAPI ingestion gateway]
                     |
                     v
   [PDF extract + chunk + local CPU embeddings (SentenceTransformer)]
                     |
                     v
        [Qdrant Cloud collection upsert (vector + payload)]
                     |
                     v
      [POST /api/v1/ask -> query enters autonomous RAG loop]
                     |
                     v
        [Retrieve top-k context from Qdrant vector space]
                     |
                     v
    [Confidence Firewall: average similarity >= threshold?]
             | Yes                               | No
             v                                   v
 [Grounded Groq/Llama 3.1 inference]   [Self-healing rewrite + re-retrieve]
             |                                   |
             +--------------------+--------------+
                                  v
         [If insufficient context -> deterministic fault frame]
                                  |
                                  v
      [Telemetry escalation via Make.com email/SMTP notification path]
```

## Technical Stack & Deep Dive

| Component | Technology Used | Architectural Purpose |
|---|---|---|
| API Gateway | FastAPI | **Asynchronous Gateway** for health, upload, and ask routes with low-latency HTTP handling. |
| Embedding Layer | SentenceTransformers (`all-MiniLM-L6-v2`) | **Local CPU-driven Embedding Generation** to keep vectorization isolated from remote embedding dependency drift. |
| Vector Engine | Qdrant Cloud Cluster | **Vector Space Engine** for semantic indexing, top-k similarity search, and payload-backed context retrieval. |
| Inference Layer | Groq API / Llama 3.1 | **Context-Informed Inference Generator** constrained by retrieval-grounded prompting and deterministic fallback contract. |
| Event Orchestration | Make.com | **Autonomous Event Plane** for folder watch automation, API chaining, and failure notification workflows. |

## Infrastructure & Deployed Lifecycle

### Phase 1: Ingestion Plane
Asynchronous document detection is initiated by edge synchronization events (e.g., watched folder updates via Make.com + cloud drive modules). Triggered documents are forwarded to `POST /api/v1/upload` as multipart streams.

### Phase 2: Local Vectorization Core
Inside the containerized serverless runtime, PDF text is extracted, chunked, and embedded locally using `sentence-transformers`. This keeps vector generation inside the deployment boundary and minimizes external data exposure during embedding.

### Phase 3: Mathematical Guardrails
Incoming query retrieval is evaluated against a confidence score threshold (average similarity policy). This acts as a pre-inference reliability gate to block low-quality context from contaminating generation.

### Phase 4: Self-Healing Telemetry Plane
Low-confidence retrieval triggers autonomous rewrite + re-retrieval. If the model still reports insufficient context, SHBRAG emits a deterministic failure state and escalates workflow telemetry through email/SMTP vectors in the automation plane.

> **Critical reliability rule:** SHBRAG returns `status: failed` with explicit hallucination prevention messaging when context cannot be validated.

## Step-by-Step Deployment & Environment Topology

### 1) Secure environment key topology
Configure secrets as environment variables (never hardcode):

- `GROQ_API_KEY`
- `QDRANT_URL`
- `QDRANT_API_KEY`
- `HF_API_KEY`

Example (`.env` for local development):

```bash
GROQ_API_KEY=your_groq_key
QDRANT_URL=https://your-qdrant-cluster-url
QDRANT_API_KEY=your_qdrant_key
HF_API_KEY=placeholder_or_hf_key
```

### 2) Run locally with Uvicorn

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 3) Deploy to Hugging Face Spaces (Docker)

1. Create a new Hugging Face Space with **Docker** SDK.
2. Add repository secrets in Space settings: `GROQ_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `HF_API_KEY`.
3. Ensure the project `Dockerfile` is at repository root and references `main:app` runtime.
4. Push code to the connected branch; Hugging Face Spaces will build and deploy automatically.
5. Validate health endpoint after deployment: `GET /health`.

> **Deployment constraint:** Startup will fail fast if required keys are missing due to import-time configuration validation.

## API Endpoints Verification Test Harness

### Route: `POST /api/v1/upload`
- Purpose: multipart/form-data document ingestion and vectorization trigger.
- Expected response contract: success object with ingestion metadata.

**Local verification (PowerShell):**

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/api/v1/upload" `
  -Form @{ file = Get-Item ".\\sample.pdf" }
```

**Remote verification (PowerShell):**

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "https://engrahmedrehan-shbrag-autonomous-api.hf.space/api/v1/upload" `
  -Form @{ file = Get-Item ".\\sample.pdf" }
```

### Route: `POST /api/v1/ask`
- Purpose: JSON query processing with grounded answer generation or deterministic fault frame.
- Expected responses:
  - `status: success` + answer payload
  - `status: failed` + hallucination prevention reason

**Local verification (PowerShell):**

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/api/v1/ask" `
  -ContentType "application/json" `
  -Body '{"query":"What does SHBRAG do?"}'
```

**Remote verification (PowerShell):**

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "https://engrahmedrehan-shbrag-autonomous-api.hf.space/api/v1/ask" `
  -ContentType "application/json" `
  -Body '{"query":"What does SHBRAG do?"}'
```
