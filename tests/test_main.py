from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

os.environ.setdefault("QDRANT_API_KEY", "dummy")
os.environ.setdefault("GROQ_API_KEY", "dummy")
os.environ.setdefault("HF_API_KEY", "dummy")
os.environ.setdefault("QDRANT_URL", "https://example.com")

import main


class ApiEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)

    @patch("main.process_and_upload_pdf", return_value=4)
    def test_upload_endpoint_processes_file_and_returns_chunk_count(self, mock_process):
        response = self.client.post(
            "/api/v1/upload",
            files={"file": ("doc.pdf", b"%PDF-1.4 sample", "application/pdf")},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": "success",
                "message": "File processed and uploaded successfully.",
                "chunk_count": 4,
            },
        )
        mock_process.assert_called_once()
        uploaded_path = Path(mock_process.call_args.args[0])
        self.assertEqual(uploaded_path.parent.name, "data")
        self.assertFalse(uploaded_path.exists())

    @patch("main.autonomous_rag_pipeline", return_value={"status": "success", "answer": "ok"})
    def test_ask_endpoint_returns_pipeline_payload_directly(self, mock_pipeline):
        response = self.client.post("/api/v1/ask", json={"query": "What is SHBRAG?"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "success", "answer": "ok"})
        mock_pipeline.assert_called_once_with("What is SHBRAG?")


if __name__ == "__main__":
    unittest.main()
