from scripts.analyze_logs import load_records
from unittest.mock import patch
import rag

TEST_LOG_PATH = rag.BASE_DIR / "logs" / "query_logs_mock.jsonl"




if __name__ == "__main__":
    before_count = 0

    if TEST_LOG_PATH.exists():
        before_count = len(load_records(TEST_LOG_PATH))

    with patch(
            "rag.generate_answer",
            side_effect=[None, "", " ", "这是一条模拟回答"]
    ):
        with patch("rag.LOG_PATH", TEST_LOG_PATH):
            with patch("rag.OpenAI"):
                with patch(
                    "builtins.input",
                    side_effect=[
                        "RAG是什么？",
                        "RAG是什么？",
                        "RAG是什么？",
                        "RAG是什么？",
                        "exit"
                    ]
                ):
                    rag.main()

    records = load_records(TEST_LOG_PATH)
    new_records = records[before_count:]

    actual_statuses = []
    for record in new_records:
        actual_statuses.append(record["status"])

    expected_statuses = [
        "no_answer", "no_answer", "no_answer", "answered"
    ]

    assert actual_statuses == expected_statuses, (
        f"状态不符合预期，实际结果：{actual_statuses}"
    )

    print("测试通过：本次新增四条记录，状态顺序正确")