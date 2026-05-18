from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src import retrieve


class RetrieveTests(unittest.TestCase):
    @patch("src.retrieve.get_hf_embedding", return_value=[0.1, 0.2, 0.3])
    @patch("src.retrieve.qdrant_client.search")
    def test_retrieve_context_returns_text_and_score(self, mock_search, mock_embedding):
        mock_search.return_value = [
            SimpleNamespace(payload={"text": "chunk one"}, score=0.95),
            SimpleNamespace(payload={"text": "chunk two"}, score=0.81),
        ]

        results = retrieve.retrieve_context("What is SHBRAG?", top_k=2)

        self.assertEqual(
            results,
            [
                {"text": "chunk one", "score": 0.95},
                {"text": "chunk two", "score": 0.81},
            ],
        )
        mock_embedding.assert_called_once_with("What is SHBRAG?")
        mock_search.assert_called_once_with(
            collection_name=retrieve.COLLECTION_NAME,
            query_vector=[0.1, 0.2, 0.3],
            limit=2,
            with_payload=True,
        )

    @patch("src.retrieve.qdrant_client.search")
    def test_retrieve_context_with_non_positive_top_k_returns_empty(self, mock_search):
        self.assertEqual(retrieve.retrieve_context("query", top_k=0), [])
        mock_search.assert_not_called()


class GenerateAnswerTests(unittest.TestCase):
    @patch("src.retrieve.groq_client.chat.completions.create")
    def test_generate_answer_uses_required_prompt_and_model_settings(self, mock_create):
        mock_create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Answer text"))]
        )

        answer = retrieve.generate_answer(
            "What does SHBRAG do?",
            [{"text": "SHBRAG is a self-healing autonomous RAG infrastructure.", "score": 0.9}],
        )

        self.assertEqual(answer, "Answer text")
        self.assertEqual(mock_create.call_count, 1)

        kwargs = mock_create.call_args.kwargs
        self.assertEqual(kwargs["model"], "llama3-8b-8192")
        self.assertEqual(kwargs["temperature"], 0.0)

        system_prompt = kwargs["messages"][0]["content"]
        self.assertIn(
            "Act as an expert research assistant. Answer the user's question using ONLY the provided context. "
            "If the context does not contain the answer, explicitly state: 'INSUFFICIENT_CONTEXT'.",
            system_prompt,
        )


if __name__ == "__main__":
    unittest.main()
