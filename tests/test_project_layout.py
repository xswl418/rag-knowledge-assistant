"""Offline checks for package imports, entrypoints, and repository resources."""

import io
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from contextlib import redirect_stdout

from rag_assistant import agent, rag
from rag_assistant.document_task import prepare_document_batches
from scripts.evaluate_documents import CASES_PATH, load_case_chunks, load_eval_cases


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ProjectLayoutTests(unittest.TestCase):
    def run_isolated(self, code, cwd):
        environment = os.environ.copy()
        for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL"):
            environment.pop(name, None)
        result = subprocess.run(
            [sys.executable, "-I", "-B", "-c", code, str(PROJECT_ROOT)],
            cwd=cwd,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_runtime_imports_need_no_credentials_or_model(self):
        self.run_isolated(
            """
import importlib
import sys
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
with patch("openai.OpenAI", side_effect=AssertionError("Unexpected client")):
    for name in (
        "rag", "agent", "api_client", "conversation",
        "document_task", "streaming", "tools",
    ):
        importlib.import_module("rag_assistant." + name)
assert "sentence_transformers" not in sys.modules
""",
            PROJECT_ROOT,
        )

    def test_model_name_can_be_overridden_without_credentials(self):
        environment = os.environ.copy()
        for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL"):
            environment.pop(name, None)
        environment["LLM_MODEL_NAME"] = "provider-fast-model"
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                """
import sys
sys.path.insert(0, sys.argv[1])
from rag_assistant import rag
assert rag.LLM_MODEL_NAME == "provider-fast-model"
""",
                str(PROJECT_ROOT),
            ],
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_shared_paths_remain_at_repository_root(self):
        self.assertEqual(rag.BASE_DIR, PROJECT_ROOT)
        self.assertEqual(rag.FILE_PATH, PROJECT_ROOT / "data/samples/knowledge_base.txt")
        self.assertEqual(rag.LOG_PATH, PROJECT_ROOT / "logs/query_logs.jsonl")
        self.assertEqual(agent.AGENT_LOG_PATH, PROJECT_ROOT / "logs/agent_logs.jsonl")

    def test_local_assets_and_launcher_target_exist(self):
        for relative in (
            "web_app/app.py", "web_app/assets/styles.css", "web_app/assets/logo.svg",
            ".streamlit/config.toml", "requirements.txt",
        ):
            with self.subTest(path=relative):
                self.assertTrue((PROJECT_ROOT / relative).is_file())
        launcher = (PROJECT_ROOT / "start_web.bat").read_text(encoding="utf-8")
        self.assertIn('streamlit run "web_app\\app.py"', launcher)

    def test_all_python_sources_compile(self):
        for directory in ("rag_assistant", "web_app", "scripts", "tests"):
            for path in (PROJECT_ROOT / directory).rglob("*.py"):
                with self.subTest(path=str(path.relative_to(PROJECT_ROOT))):
                    compile(path.read_text(encoding="utf-8"), str(path), "exec")

    def test_sample_documents_parse_with_original_sources(self):
        for path in (PROJECT_ROOT / "data/samples").iterdir():
            if path.suffix.lower() not in (".txt", ".docx", ".pdf"):
                continue
            with self.subTest(document=path.name), redirect_stdout(io.StringIO()):
                chunks = rag.load_document_chunks(
                    path, max_chunk_size=rag.MAX_CHUNK_SIZE,
                    overlap=rag.CHUNK_OVERLAP,
                )
                self.assertTrue(chunks)
                self.assertTrue(all(c["source"] == path.name for c in chunks))
                self.assertTrue(all(0 < len(c["content"]) <= rag.MAX_CHUNK_SIZE for c in chunks))

    def test_pdf_blank_page_does_not_renumber_sources(self):
        with redirect_stdout(io.StringIO()):
            chunks = rag.load_document_chunks(
                PROJECT_ROOT / "data/samples/library_pages_test.pdf",
                max_chunk_size=rag.MAX_CHUNK_SIZE, overlap=rag.CHUNK_OVERLAP,
            )
        self.assertEqual({c["page_number"] for c in chunks}, {1, 3})

    def test_evaluation_cases_resolve_and_parse_documents(self):
        cases = load_eval_cases(CASES_PATH)
        self.assertEqual(len(cases), 12)
        self.assertEqual(len({case["id"] for case in cases}), 12)
        for case in cases:
            with self.subTest(case=case["id"]), redirect_stdout(io.StringIO()):
                self.assertTrue(load_case_chunks(case))

    def test_long_document_batches_preserve_all_chunks(self):
        for name in ("summary_long_test.txt", "extraction_long_test.txt"):
            with self.subTest(document=name), redirect_stdout(io.StringIO()):
                chunks = rag.load_document_chunks(
                    PROJECT_ROOT / "data/samples" / name,
                    max_chunk_size=rag.MAX_CHUNK_SIZE, overlap=rag.CHUNK_OVERLAP,
                )
                batches, error = prepare_document_batches(chunks)
                self.assertIsNone(error)
                self.assertGreater(len(batches), 1)
                self.assertEqual([chunk for batch in batches for chunk in batch], chunks)

    def test_public_markdown_local_links_resolve(self):
        paths = [PROJECT_ROOT / "README.md", PROJECT_ROOT / "CHANGELOG.md"]
        paths.extend((PROJECT_ROOT / "docs").glob("*.md"))
        for path in paths:
            for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
                if "://" in target or target.startswith("#"):
                    continue
                with self.subTest(document=path.name, target=target):
                    self.assertTrue((path.parent / target.split("#", 1)[0]).exists())

    def test_web_entry_bootstraps_from_unrelated_working_directory(self):
        with TemporaryDirectory() as directory:
            self.run_isolated(
                """
from pathlib import Path
import sys
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
root = Path(sys.argv[1])
assert str(root) not in sys.path
with patch("openai.OpenAI", side_effect=AssertionError("Unexpected client")):
    app = AppTest.from_file(str(root / "web_app/app.py")).run(timeout=30)
assert not app.exception, str(app.exception)
assert "rag_assistant.agent" in sys.modules
assert "sentence_transformers" not in sys.modules
""",
                directory,
            )


if __name__ == "__main__":
    unittest.main()
