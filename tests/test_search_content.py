from unittest import TestCase
from rag_assistant.agent import build_search_content

class TestSearchContent(TestCase):
    def test_empty_results(self):
        results = []
        remaining_search_calls = 2

        actual = build_search_content(results, remaining_search_calls)
        self.assertEqual(actual["results"], [])
        self.assertIn("message", actual)

    def test_low_confidence(self):
        results = [{"chunk": "hhh", "score": 0.2, "low_confidence": True}]
        remaining_search_calls = 2

        actual = build_search_content(results, remaining_search_calls)
        self.assertEqual(actual["results"], results)
        self.assertIn("阈值", actual["message"])

    def test_normal_result(self):
        results = [{"chunks": "hhh", "score": 0.2}]
        remaining_search_calls = 2

        actual = build_search_content(results, remaining_search_calls)
        self.assertEqual(actual, results)
