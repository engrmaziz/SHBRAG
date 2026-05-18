from __future__ import annotations

import unittest
from unittest.mock import patch

from src import healing


class EvaluateConfidenceTests(unittest.TestCase):
    def test_evaluate_confidence_returns_true_for_high_average_score(self):
        self.assertTrue(healing.evaluate_confidence([{"score": 0.8}, {"score": 0.7}]))

    def test_evaluate_confidence_returns_false_for_low_average_or_empty(self):
        self.assertFalse(healing.evaluate_confidence([{"score": 0.6}, {"score": 0.5}]))
        self.assertFalse(healing.evaluate_confidence([]))


class AutonomousPipelineTests(unittest.TestCase):
    @patch("src.healing.generate_answer", return_value="Grounded answer")
    @patch("src.healing.retrieve_context")
    def test_pipeline_success_without_fallback(self, mock_retrieve, mock_generate):
        mock_retrieve.return_value = [
            {"text": "chunk one", "score": 0.8, "source": "doc-a"},
            {"text": "chunk two", "score": 0.9, "source": "doc-b"},
        ]

        result = healing.autonomous_rag_pipeline("What is SHBRAG?")

        self.assertEqual(
            result,
            {
                "status": "success",
                "answer": "Grounded answer",
                "sources_used": ["doc-a", "doc-b"],
            },
        )
        mock_retrieve.assert_called_once_with("What is SHBRAG?", top_k=3)
        mock_generate.assert_called_once()

    @patch("src.healing.generate_answer", return_value="Recovered answer")
    @patch("src.healing.retrieve_context")
    def test_pipeline_uses_fallback_retrieval_on_low_confidence(self, mock_retrieve, mock_generate):
        mock_retrieve.side_effect = [
            [{"text": "weak chunk", "score": 0.2}],
            [{"text": "strong chunk", "score": 0.9, "source_file": "paper.pdf"}],
        ]

        result = healing.autonomous_rag_pipeline("Explain SHBRAG")

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["sources_used"], ["paper.pdf"])
        self.assertEqual(mock_retrieve.call_count, 2)
        mock_retrieve.assert_any_call("Explain SHBRAG", top_k=3)
        mock_retrieve.assert_any_call(
            "Explain SHBRAG (rewrite for better semantic search)", top_k=5
        )
        mock_generate.assert_called_once_with(
            "Explain SHBRAG", [{"text": "strong chunk", "score": 0.9, "source_file": "paper.pdf", "source": "paper.pdf"}]
        )

    @patch("src.healing.generate_answer", return_value="INSUFFICIENT_CONTEXT")
    @patch("src.healing.retrieve_context", return_value=[{"text": "chunk", "score": 0.9, "source": "doc"}])
    def test_pipeline_returns_failed_status_when_context_is_insufficient(
        self, mock_retrieve, mock_generate
    ):
        result = healing.autonomous_rag_pipeline("Unanswerable query")

        self.assertEqual(
            result,
            {
                "status": "failed",
                "reason": "Hallucination prevented. No relevant data in vector space.",
                "original_query": "Unanswerable query",
            },
        )
        mock_retrieve.assert_called_once_with("Unanswerable query", top_k=3)
        mock_generate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
