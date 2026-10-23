import unittest

from rag_assistant.rag import build_sentence_chunks


class ChunkingTests(unittest.TestCase):
    def test_normal_sentence_overlap(self):
        text = "借用设备之前必须登记。设备借用期限为七天。"

        actual = build_sentence_chunks(
            text,
            max_chunk_size=18,
            overlap=4,
        )

        expected = [
            "借用设备之前必须登记。",
            "须登记。设备借用期限为七天。",
        ]

        self.assertEqual(actual, expected)

    def test_zero_overlap(self):
        text = "借用设备之前必须登记。设备借用期限为七天。"

        actual = build_sentence_chunks(
            text,
            max_chunk_size=18,
            overlap=0,
        )

        expected = [
            "借用设备之前必须登记。",
            "设备借用期限为七天。",
        ]

        self.assertEqual(actual, expected)
