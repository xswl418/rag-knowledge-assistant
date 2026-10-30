import unittest

from rag_assistant.rag import retrieve_top_k

class DefineEncode:
    def encode(self, question):
        return [1.0, 0.0]

class TestRetrieval(unittest.TestCase):
    def test_retrieve_top_k(self):
        minimum_score = 0.9
        top_k=3
        question = "几点开放"
        vectors = [[0.0,1.0], [0.6, 0.8], [0.8, 0.6]]
        chunks = [{"content":"A"}, {"content":"B"}, {"content":"C"}]
        model = DefineEncode()

        actual = retrieve_top_k(question, chunks, vectors, model, top_k, minimum_score)

        expected = [
            {"chunk": {"content": "C"}, "score": 0.8, "low_confidence": True}
        ]

        self.assertEqual(len(actual), 1)
        self.assertEqual(actual[0]["chunk"]["content"], "C")
        self.assertAlmostEqual(actual[0]["score"], 0.8, places=5)
        self.assertEqual(actual[0]["low_confidence"], True)