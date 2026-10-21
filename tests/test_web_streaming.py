"""Offline checks for the webpage's live text and final answer placement."""

import ast
import io
import json
import os
import sys
import time
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from httpx import ReadError, Request
from openai import APIConnectionError
from openai.types.chat import ChatCompletionChunk
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rag_assistant import agent, api_client, rag
from rag_assistant.conversation import build_history_messages


def text_chunk(text=None, finish=None, tool_calls=None):
    return ChatCompletionChunk(
        id="offline",
        created=0,
        model="offline",
        object="chat.completion.chunk",
        choices=[{
            "index": 0,
            "delta": {"content": text, "tool_calls": tool_calls},
            "finish_reason": finish,
        }],
    )


class FakeStream:
    def __init__(self, pieces):
        self.pieces = pieces

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def __iter__(self):
        for piece in self.pieces:
            if isinstance(piece, Exception):
                raise piece
            yield piece


@contextmanager
def fake_embedding_module():
    missing = object()
    previous = sys.modules.get("sentence_transformers", missing)
    module = SimpleNamespace(
        SentenceTransformer=MagicMock(return_value=object()),
    )
    sys.modules["sentence_transformers"] = module
    try:
        yield module
    finally:
        if previous is missing:
            sys.modules.pop("sentence_transformers", None)
        else:
            sys.modules["sentence_transformers"] = previous


class WebStreamingTests(unittest.TestCase):
    def setUp(self):
        self.upload = io.BytesIO(b"Each reader may borrow 3 books for 12 days.")
        self.upload.name = "library.txt"
        self.upload.size = len(self.upload.getvalue())
        self.chunk = {
            "source": self.upload.name,
            "chunk_id": 0,
            "paragraph_id": 1,
            "content": self.upload.getvalue().decode(),
        }
        self.client = MagicMock()
        self.client.__enter__.return_value = self.client
        self.updates = []
        markdown = DeltaGenerator.markdown

        def track_markdown(target, body, *args, **kwargs):
            self.updates.append((str(body), target._get_delta_path_str()))
            return markdown(target, body, *args, **kwargs)

        self.saved = self.enterContext(patch.object(rag, "save_record"))
        self.enterContext(patch.object(api_client, "create_api_client", return_value=self.client))
        self.enterContext(patch("streamlit.file_uploader", return_value=self.upload))
        self.document_loader = self.enterContext(patch.object(
            rag, "load_document_chunks", return_value=[self.chunk],
        ))
        self.vector_builder = self.enterContext(patch.object(
            rag, "create_embeddings", return_value=SimpleNamespace(shape=(1, 3)),
        ))
        self.enterContext(fake_embedding_module())
        self.enterContext(patch.object(DeltaGenerator, "markdown", track_markdown))
        self.enterContext(patch.dict(agent.available_tools, {
            "search_knowledge": MagicMock(return_value=[{
                "chunk": self.chunk,
                "score": 0.9,
            }]),
        }))
        self.enterContext(redirect_stdout(io.StringIO()))

    def submit(self, streams, question="How many books?"):
        self.client.chat.completions.create.side_effect = streams
        app = AppTest.from_file(str(PROJECT_ROOT / "web_app" / "app.py"))
        app.run(timeout=30)
        self.assertFalse(app.exception)
        app.text_input(key="question_input").set_value(question)
        next(button for button in app.button if button.label == "\u63d0\u95ee").click()
        app.run(timeout=30)
        self.assertFalse(app.exception)
        return app

    def test_direct_answer_streams_without_duplicate(self):
        app = self.submit([FakeStream([
            text_chunk("Hello "), text_chunk("there."), text_chunk(finish="stop"),
        ])])
        self.assertIn("Hello ", [body for body, _ in self.updates])
        final_paths = [path for body, path in self.updates if body == "Hello there."]
        self.assertEqual(len(final_paths), 2)
        # A direct response initially uses the preface slot, then the final slot.
        self.assertEqual([m.value for m in app.markdown].count("Hello there."), 1)
        self.saved.assert_called_once()
        app.run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(self.client.chat.completions.create.call_count, 1)
        self.saved.assert_called_once()

    def test_retrieval_preserves_preface_and_streams_final_in_place(self):
        tool = {
            "index": 0,
            "id": "search-1",
            "type": "function",
            "function": {
                "name": "search_knowledge",
                "arguments": '{"question": "How many books?"}',
            },
        }
        app = self.submit([
            FakeStream([
                text_chunk("I will "), text_chunk("search first."),
                text_chunk(tool_calls=[tool]), text_chunk(finish="tool_calls"),
            ]),
            FakeStream([
                text_chunk("3 books"), text_chunk(", 12 days."),
                text_chunk(finish="stop"),
            ]),
        ])
        bodies = [body for body, _ in self.updates]
        self.assertIn("I will ", bodies)
        self.assertIn("3 books", bodies)
        self.assertNotIn("I will search first.3 books", bodies)
        final_paths = [path for body, path in self.updates if body == "3 books, 12 days."]
        self.assertEqual(len(final_paths), 2)
        self.assertEqual(final_paths[0], final_paths[1])
        record = app.session_state["chat_records"][0]
        self.assertEqual(record["retrieval_preface"], "I will search first.")
        self.assertEqual(record["answer"], "3 books, 12 days.")
        self.assertEqual(build_history_messages([record])[-1]["content"], record["answer"])
        self.saved.assert_called_once()
        app.run(timeout=30)
        self.assertFalse(app.exception)
        bodies = [m.value for m in app.markdown]
        self.assertEqual(bodies.count(record["retrieval_preface"]), 1)
        self.assertEqual(bodies.count(record["answer"]), 1)
        self.assertEqual(self.client.chat.completions.create.call_count, 2)
        self.saved.assert_called_once()
        app.button(key="clear_conversation").click().run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["chat_records"], [])

    def test_interrupted_stream_clears_partial_text(self):
        app = self.submit([FakeStream([
            text_chunk("Unfinished answer"), ReadError("offline interruption"),
        ])])
        self.assertEqual(app.session_state["chat_records"][0]["status"], "api_error")
        self.assertNotIn("Unfinished answer", [m.value for m in app.markdown])
        self.assertTrue(app.error)
        self.saved.assert_called_once()

    def test_abnormal_finish_clears_partial_text(self):
        app = self.submit([FakeStream([
            text_chunk("Truncated answer"), text_chunk(finish="length"),
        ])])
        self.assertEqual(app.session_state["chat_records"][0]["status"], "no_answer")
        self.assertNotIn("Truncated answer", [m.value for m in app.markdown])
        self.assertTrue(app.warning)
        self.saved.assert_called_once()

    def summary_selection(self, preface=True):
        pieces = []
        if preface:
            pieces.extend([text_chunk("I will "), text_chunk("summarize first.")])
        pieces.extend([
            text_chunk(tool_calls=[{
                "index": 0,
                "id": "summary-1",
                "type": "function",
                "function": {"name": "summarize_document", "arguments": "{}"},
            }]),
            text_chunk(finish="tool_calls"),
        ])
        return FakeStream(pieces)

    def use_long_document(self, batch_count=2):
        self.upload.seek(0)
        self.upload.truncate(0)
        self.upload.write(b"x" * (7000 * batch_count))
        self.upload.seek(0)
        self.upload.size = len(self.upload.getvalue())
        self.document_loader.return_value = [
            dict(self.chunk, chunk_id=index, paragraph_id=index + 1, content="x" * 7000)
            for index in range(batch_count)
        ]

    def check_summary_record(self, app, status, model_calls):
        record = app.session_state["chat_records"][0]
        self.assertEqual(record["status"], status)
        self.assertEqual(record["agent_model_call_count"], 1)
        self.assertEqual(record["model_call_count"], model_calls)
        self.assertEqual(record["tool_call_count"], 1)
        self.assertEqual(len(record["model_timings"]), model_calls)
        self.assertEqual(self.client.chat.completions.create.call_count, model_calls)
        self.assertTrue(all(
            call.kwargs["stream"] for call in self.client.chat.completions.create.call_args_list
        ))
        self.vector_builder.assert_not_called()
        self.saved.assert_called_once()
        if status != "answered":
            self.assertIsNone(record["answer"])
            self.assertNotIn("summary_preface", record)
            bodies = [m.value for m in app.markdown]
            self.assertNotIn("I will summarize first.", bodies)
            self.assertNotIn("Partial summary", bodies)
            self.assertTrue(app.warning)
        return record

    def test_single_summary_streams_in_place_and_preserves_preface(self):
        app = self.submit([
            self.summary_selection(),
            FakeStream([
                text_chunk("Single "), text_chunk("summary"), text_chunk(finish="stop"),
            ]),
        ], question="Summarize the document.")
        record = self.check_summary_record(app, "answered", 2)
        self.assertEqual(record["summary_preface"], "I will summarize first.")
        self.assertEqual(record["answer"], "Single summary")
        self.assertEqual(record["summary_input"]["batch_count"], 1)
        bodies = [body for body, _ in self.updates]
        self.assertIn("I will ", bodies)
        self.assertIn("Single ", bodies)
        self.assertNotIn("I will summarize first.Single ", bodies)
        final_paths = [path for body, path in self.updates if body == "Single summary"]
        self.assertEqual(len(final_paths), 2)
        self.assertEqual(final_paths[0], final_paths[1])
        self.assertEqual(build_history_messages([record])[-1]["content"], "Single summary")
        app.run(timeout=30)
        self.assertFalse(app.exception)
        bodies = [m.value for m in app.markdown]
        self.assertEqual(bodies.count(record["summary_preface"]), 1)
        self.assertEqual(bodies.count(record["answer"]), 1)
        self.saved.assert_called_once()
        self.assertEqual(self.client.chat.completions.create.call_count, 2)
        app.button(key="clear_conversation").click().run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["chat_records"], [])

    def test_multibatch_summary_only_streams_final_merge(self):
        self.use_long_document()
        app = self.submit([
            self.summary_selection(),
            FakeStream([text_chunk("Internal batch one"), text_chunk(finish="stop")]),
            FakeStream([text_chunk("Internal batch two"), text_chunk(finish="stop")]),
            FakeStream([
                text_chunk("Final "), text_chunk("summary"), text_chunk(finish="stop"),
            ]),
        ], question="Summarize the document.")
        record = self.check_summary_record(app, "answered", 4)
        self.assertEqual(record["answer"], "Final summary")
        self.assertEqual(record["summary_preface"], "I will summarize first.")
        self.assertEqual(record["summary_input"]["batch_count"], 2)
        self.assertEqual([entry["stage"] for entry in record["model_timings"]], [
            "agent_model", "summary_model", "summary_model", "summary_merge_model",
        ])
        bodies = [body for body, _ in self.updates]
        self.assertIn("Final ", bodies)
        self.assertNotIn("Internal batch one", bodies)
        self.assertNotIn("Internal batch two", bodies)
        final_paths = [path for body, path in self.updates if body == "Final summary"]
        self.assertEqual(len(final_paths), 2)
        self.assertEqual(final_paths[0], final_paths[1])
        app.run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual([m.value for m in app.markdown].count("Final summary"), 1)
        self.saved.assert_called_once()
        self.assertEqual(self.client.chat.completions.create.call_count, 4)

    def test_summary_without_preface_streams_body(self):
        app = self.submit([
            self.summary_selection(preface=False),
            FakeStream([text_chunk("Summary"), text_chunk(finish="stop")]),
        ], question="Summarize the document.")
        record = self.check_summary_record(app, "answered", 2)
        self.assertEqual(record["answer"], "Summary")
        self.assertNotIn("summary_preface", record)
        self.assertEqual([m.value for m in app.markdown].count("Summary"), 1)

    def test_single_summary_interruption_clears_partial_text(self):
        app = self.submit([
            self.summary_selection(),
            FakeStream([text_chunk("Partial summary"), ReadError("offline")]),
        ], question="Summarize the document.")
        self.check_summary_record(app, "summary_error", 2)

    def test_intermediate_batch_failure_does_not_start_merge(self):
        self.use_long_document()
        app = self.submit([
            self.summary_selection(),
            FakeStream([text_chunk("Internal batch one"), text_chunk(finish="stop")]),
            FakeStream([text_chunk("Partial summary"), ReadError("offline")]),
        ], question="Summarize the document.")
        self.check_summary_record(app, "summary_error", 3)
        bodies = [body for body, _ in self.updates]
        self.assertNotIn("Internal batch one", bodies)
        self.assertNotIn("Partial summary", bodies)

    def test_merge_interruption_clears_partial_text(self):
        self.use_long_document()
        app = self.submit([
            self.summary_selection(),
            FakeStream([text_chunk("Internal batch one"), text_chunk(finish="stop")]),
            FakeStream([text_chunk("Internal batch two"), text_chunk(finish="stop")]),
            FakeStream([text_chunk("Partial summary"), ReadError("offline")]),
        ], question="Summarize the document.")
        self.check_summary_record(app, "summary_error", 4)
        self.assertIn("Partial summary", [body for body, _ in self.updates])

    def test_summary_abnormal_finish_clears_partial_text(self):
        app = self.submit([
            self.summary_selection(),
            FakeStream([text_chunk("Partial summary"), text_chunk(finish="length")]),
        ], question="Summarize the document.")
        self.check_summary_record(app, "summary_error", 2)

    def test_summary_rejects_plan_beyond_limit(self):
        self.use_long_document(batch_count=11)
        app = self.submit([
            self.summary_selection(),
        ], question="Summarize the document.")
        self.check_summary_record(app, "summary_error", 1)

    def extraction_selection(self, preface=True):
        pieces = [text_chunk("I will extract first.")] if preface else []
        pieces.extend([
            text_chunk(tool_calls=[{
                "index": 0,
                "id": "extract-1",
                "type": "function",
                "function": {
                    "name": "extract_document",
                    "arguments": '{"fields": ["limit", "days", "fee"]}',
                },
            }]),
            text_chunk(finish="tool_calls"),
        ])
        return FakeStream(pieces)

    def extraction_response(self, index=0, finish="stop", invalid_json=False, bad_evidence=False):
        limit = "5 books" if index == 2 else "3 books"
        evidence = f"Readers may borrow {limit}."
        if bad_evidence:
            evidence = "This quotation is absent."
        days = None if index == 1 else "12 days"
        data = {"items": [
            {"field": "limit", "value": limit, "evidence": evidence},
            {
                "field": "days",
                "value": days,
                "evidence": "Loan period is 12 days." if days else None,
            },
            {"field": "fee", "value": None, "evidence": None},
        ]}
        content = "{invalid" if invalid_json else json.dumps(data)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=content), finish_reason=finish,
        )])

    def use_extraction_document(self, count=1, pdf=False):
        self.upload.name = "guide.pdf" if pdf else "guide.txt"
        chunks = []
        for index in range(count):
            content = (
                f"Readers may borrow {'5' if index == 2 else '3'} books. "
                "Loan period is 12 days. " + "x" * (7000 if count > 1 else 30)
            )
            chunk = {
                "source": self.upload.name, "chunk_id": index, "content": content,
            }
            chunk["page_number" if pdf else "paragraph_id"] = index + 1
            chunks.append(chunk)
        self.document_loader.return_value = chunks
        self.upload.seek(0)
        self.upload.truncate(0)
        self.upload.write("\n".join(c["content"] for c in chunks).encode())
        self.upload.seek(0)
        self.upload.size = len(self.upload.getvalue())

    def submit_extraction(self, responses, preface=True):
        self.extraction_events = []
        self.progress_updates = []
        real_run = agent.run_agent
        real_progress = st.progress

        def tracked_run(*args, **kwargs):
            callback = kwargs["on_extraction_progress"]

            def report(event):
                self.extraction_events.append(dict(event))
                callback(event)

            kwargs["on_extraction_progress"] = report
            return real_run(*args, **kwargs)

        def tracked_progress(value, *args, **kwargs):
            self.progress_updates.append((value, kwargs.get("text", "")))
            return real_progress(value, *args, **kwargs)

        self.enterContext(patch.object(agent, "run_agent", side_effect=tracked_run))
        self.enterContext(patch.object(st, "progress", tracked_progress))
        return self.submit(
            [self.extraction_selection(preface), *responses],
            question="Extract limit, days and fee.",
        )

    def check_extraction_record(self, app, status, calls):
        record = app.session_state["chat_records"][0]
        self.assertEqual(record["status"], status)
        self.assertEqual(record["agent_model_call_count"], 1)
        self.assertEqual(record["model_call_count"], calls)
        self.assertEqual(len(record["model_timings"]), calls)
        self.assertEqual(self.client.chat.completions.create.call_count, calls)
        self.assertEqual(record["tool_call_count"], 1)
        self.assertEqual(record["extraction_fields"], ["limit", "days", "fee"])
        self.assertEqual(len(app.get("progress")), 0)
        self.vector_builder.assert_not_called()
        self.saved.assert_called_once()
        if status == "answered":
            self.assertEqual(json.loads(record["answer"]), record["extraction_result"])
            self.assertNotIn(record["answer"], [m.value for m in app.markdown])
        else:
            self.assertIsNone(record["answer"])
            self.assertIsNone(record["extraction_result"])
            self.assertNotIn("extraction_preface", record)
            self.assertNotIn("I will extract first.", [m.value for m in app.markdown])
            self.assertNotIn("3 books", [t.value for t in app.text])
            self.assertTrue(app.error)
        return record

    def test_single_extraction_preserves_preface_and_reruns_once(self):
        self.use_extraction_document()
        app = self.submit_extraction([self.extraction_response()])
        record = self.check_extraction_record(app, "answered", 2)
        self.assertEqual(record["extraction_preface"], "I will extract first.")
        self.assertEqual(record["extraction_input"]["batch_count"], 1)
        self.assertEqual([v["status"] for v in record["extraction_result"]["items"]], [
            "single", "single", "missing",
        ])
        self.assertIn("3 books", [t.value for t in app.text])
        self.assertIn("未找到", [t.value for t in app.text])
        self.assertIn("guide.txt，分组 1", [t.value for t in app.text])
        self.assertEqual([v for v, _ in self.progress_updates], [0.0, 1.0])
        self.assertEqual(build_history_messages([record])[-1]["content"], record["answer"])
        app.run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual([m.value for m in app.markdown].count(record["extraction_preface"]), 1)
        self.assertEqual([t.value for t in app.text].count("3 books"), 1)
        self.assertEqual(self.client.chat.completions.create.call_count, 2)
        self.saved.assert_called_once()
        app.button(key="clear_conversation").click().run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["chat_records"], [])

    def test_multibatch_extraction_pairs_candidates_with_pdf_evidence(self):
        self.use_extraction_document(3, pdf=True)
        app = self.submit_extraction([self.extraction_response(i) for i in range(3)])
        record = self.check_extraction_record(app, "answered", 4)
        self.assertEqual(record["extraction_input"]["batch_count"], 3)
        self.assertEqual(record["extraction_input"]["model_calls_required"], 3)
        self.assertEqual(len(record["extraction_input"]["batch_char_counts"]), 3)
        limit, days, missing = record["extraction_result"]["items"]
        self.assertEqual([limit["status"], days["status"], missing["status"]], [
            "multiple", "single", "missing",
        ])
        self.assertEqual([o["value"] for o in limit["options"]], ["3 books", "5 books"])
        self.assertEqual(len(limit["options"][0]["evidences"]), 2)
        field = app.get_by_key("extraction-item-0-0")
        texts = [t.value for t in field.text]
        self.assertEqual(texts, [
            "3 books", "Readers may borrow 3 books.", "guide.pdf，第 1 页",
            "Readers may borrow 3 books.", "guide.pdf，第 2 页",
            "5 books", "Readers may borrow 5 books.", "guide.pdf，第 3 页",
        ])
        self.assertTrue(any(
            "多项结果，待核对" in m.value for m in field.markdown
        ))
        self.assertEqual(
            [(e["stage"], e["completed"], e["total"]) for e in self.extraction_events],
            [(stage, completed, 3) for index in range(3)
             for stage, completed in [("batch_started", index), ("batch_completed", index + 1)]],
        )
        self.assertEqual([v for v, _ in self.progress_updates], [0, 1/3, 1/3, 2/3, 2/3, 1])
        calls = self.client.chat.completions.create.call_args_list
        self.assertTrue(calls[0].kwargs["stream"])
        self.assertTrue(all(not c.kwargs.get("stream") for c in calls[1:]))
        app.run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual([t.value for t in app.text].count("5 books"), 1)
        self.assertEqual(self.client.chat.completions.create.call_count, 4)
        self.saved.assert_called_once()

    def test_extraction_without_preface(self):
        self.use_extraction_document()
        app = self.submit_extraction([self.extraction_response()], preface=False)
        record = self.check_extraction_record(app, "answered", 2)
        self.assertNotIn("extraction_preface", record)
        self.assertIn("3 books", [t.value for t in app.text])

    def assert_second_extraction_failure(self, response):
        self.use_extraction_document(3)
        app = self.submit_extraction([self.extraction_response(), response])
        self.check_extraction_record(app, "extraction_error", 3)
        self.assertEqual(
            [(e["stage"], e["completed"]) for e in self.extraction_events],
            [("batch_started", 0), ("batch_completed", 1), ("batch_started", 1)],
        )
        self.assertEqual([v for v, _ in self.progress_updates], [0, 1/3, 1/3])

    def test_extraction_invalid_json_clears_progress_and_partial_result(self):
        self.assert_second_extraction_failure(self.extraction_response(1, invalid_json=True))

    def test_extraction_connection_failure_clears_progress_and_partial_result(self):
        self.assert_second_extraction_failure(APIConnectionError(
            request=Request("POST", "https://offline.invalid"),
        ))

    def test_extraction_abnormal_finish_clears_progress_and_partial_result(self):
        self.assert_second_extraction_failure(self.extraction_response(1, finish="length"))

    def test_extraction_invalid_evidence_clears_progress_and_partial_result(self):
        self.assert_second_extraction_failure(self.extraction_response(1, bad_evidence=True))

    def test_extraction_rejects_plan_beyond_limit(self):
        self.use_extraction_document(11)
        app = self.submit_extraction([])
        self.check_extraction_record(app, "extraction_error", 1)
        self.assertEqual(self.progress_updates, [])
        self.assertEqual(self.extraction_events, [])

    def test_extraction_budget_refusal_keeps_plan_without_progress(self):
        self.use_extraction_document(3)
        real_tool = agent.available_tools["extract_document"]

        def no_budget(**kwargs):
            kwargs["remaining_model_calls"] = 0
            return real_tool(**kwargs)

        self.enterContext(patch.dict(agent.available_tools, {"extract_document": no_budget}))
        app = self.submit_extraction([])
        record = self.check_extraction_record(app, "extraction_error", 1)
        self.assertEqual(record["extraction_input"]["model_calls_required"], 3)
        self.assertEqual(self.progress_updates, [])
        self.assertEqual(self.extraction_events, [])

    def test_extraction_legacy_record_still_displays_value_and_location(self):
        self.use_extraction_document()
        app = self.submit_extraction([self.extraction_response()])
        self.check_extraction_record(app, "answered", 2)
        record = app.session_state["chat_records"][0]
        record["extraction_result"] = {"items": [{
            "field": "legacy", "value": "old value", "evidence": "old quotation",
            "sources": [{"source": "old.pdf", "page_number": 8}],
        }]}
        app.session_state["chat_records"] = [record]
        app.run(timeout=30)
        self.assertFalse(app.exception)
        self.assertIn("old value", [t.value for t in app.text])
        self.assertIn("old quotation", [t.value for t in app.text])
        self.assertIn("old.pdf，第 8 页", [t.value for t in app.text])
        self.assertEqual(self.client.chat.completions.create.call_count, 2)
        self.saved.assert_called_once()


class WebEmbeddingConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = PROJECT_ROOT / "web_app" / "app.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "load_embedding_model"
        )
        # Test the actual loader without running the page or its resource cache.
        function.decorator_list = []
        namespace = {"os": os, "time": time}
        exec(compile(
            ast.Module(body=[function], type_ignores=[]), str(path), "exec",
        ), namespace)
        cls.load_embedding_model = staticmethod(namespace["load_embedding_model"])

    def assert_download_setting(self, value, local_files_only):
        with (
            patch.dict(os.environ) as environment,
            fake_embedding_module() as module,
            redirect_stdout(io.StringIO()),
        ):
            if value is None:
                environment.pop("EMBEDDING_ALLOW_DOWNLOAD", None)
            else:
                environment["EMBEDDING_ALLOW_DOWNLOAD"] = value
            result = self.load_embedding_model("offline-model")
            module.SentenceTransformer.assert_called_once_with(
                "offline-model", local_files_only=local_files_only,
            )
            self.assertIs(result, module.SentenceTransformer.return_value)

    def test_embedding_default_only_reads_local_files(self):
        self.assert_download_setting(None, True)

    def test_embedding_false_only_reads_local_files(self):
        self.assert_download_setting("false", True)

    def test_embedding_true_allows_download(self):
        self.assert_download_setting("true", False)

    def test_embedding_true_is_case_insensitive(self):
        self.assert_download_setting("TRUE", False)


if __name__ == "__main__":
    unittest.main()
