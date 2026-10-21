import json
from rag_assistant import rag
from pathlib import Path
from rag_assistant.agent import run_agent
from rag_assistant.api_client import create_api_client
from datetime import datetime
from uuid import uuid4

BASE_DIR = Path(__file__).resolve().parent.parent
CASES_PATH = BASE_DIR / "data" / "eval_cases.json"
EVAL_RESULTS_PATH = BASE_DIR / "logs" / "eval_results.jsonl"


def load_eval_cases(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        cases = json.load(file)

    if not isinstance(cases, list):
        raise ValueError("评测数据最外层必须是列表。")

    if not cases:
        raise ValueError("评测用例列表不能为空。")

    required_fields = [
        "id",
        "document",
        "task_type",
        "question",
        "expected_points",
    ]

    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"第{index}条用例必须是字典。")

        for field in required_fields:
            if field not in case:
                raise ValueError(f"第{index}条用例缺少字段：{field}")

        text_fields = ["id", "document", "task_type", "question"]

        for field in text_fields:
            value = case[field]

            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"第{index}条用例的{field}必须是非空文字。")

        points = case["expected_points"]

        if not isinstance(points, list) or not points:
            raise ValueError(f"第{index}条用例的expected_points必须是非空列表。")

        for point_index, point in enumerate(points, start=1):
            if not isinstance(point, str) or not point.strip():
                raise ValueError(f"第{index}条用例的第{point_index}条预期必须是非空文字。")

        history = case.get("history_messages", [])

        if not isinstance(history, list) :
            raise ValueError(f"第{index}条用例的history_messages必须是列表。")

        for message_index, message in enumerate(history, start=1):
            if not isinstance(message, dict):
                raise ValueError(f"第{index}条用例的第{message_index}条历史消息必须是字典。")

            if message.get("role") not in ["user", "assistant"]:
                raise ValueError(f"第{index}条用例的第{message_index}条历史角色必须是user或assistant。")

            content = message.get("content")

            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"第{index}条用例的第{message_index}条历史正文必须是非空文字。")

    return cases

def load_case_chunks(case):
    document_path = Path(BASE_DIR) / case["document"]

    chunks = rag.load_document_chunks(
        document_path,
        max_chunk_size=rag.MAX_CHUNK_SIZE,
        overlap=rag.CHUNK_OVERLAP,
    )

    return chunks

def select_case(cases, case_id):
    for case in cases:
        if case["id"] == case_id:
            return case

    return None

def main():
    cases = load_eval_cases(CASES_PATH)
    print("评测用例数：", len(cases))

    selection = input("请输入编号，用空格分隔；all运行全部：").strip()

    if selection != "all":
        selected_cases = []

        for case_id in selection.split():
            selected_case =select_case(cases, case_id)

            if selected_case is None:
                print("没有找到编号:", case_id)
                return

            if selected_case not in selected_cases:
                selected_cases.append(selected_case)

        if not selected_cases:
            print("没有选择用例。")
            return

        cases = selected_cases

    print("本次运行用例数:", len(cases))

    evaluation_context = {
        "batch_id": uuid4().hex,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": {
            "model": rag.LLM_MODEL_NAME,
            "embedding_model": rag.EMBEDDING_MODEL_NAME,
            "top_k": rag.TOP_K,
            "minimum_score": rag.MINIMUM_SCORE,
            "max_chunk_size": rag.MAX_CHUNK_SIZE,
            "chunk_overlap": rag.CHUNK_OVERLAP,
        },
    }

    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer(
        rag.EMBEDDING_MODEL_NAME,
        local_files_only=True,
    )
    with create_api_client() as client:
        for case in cases:
            print("编号：", case["id"])
            print("问题：", case["question"])
            print("预期要点：", case["expected_points"])

            chunks = load_case_chunks(case)
            print("知识块数量：", len(chunks))

            chunk_vectors = rag.create_embeddings(chunks, embedding_model)
            print("向量形状：", chunk_vectors.shape)

            history_messages = case.get("history_messages", [])

            record = run_agent(
                question=case["question"],
                client=client,
                knowledge_chunks=chunks,
                chunk_vectors=chunk_vectors,
                embedding_model=embedding_model,
                history_messages=history_messages,
            )

            print("运行状态:", record["status"])
            print("实际回答:", record["answer"])

            eval_record = {
                "case_id": case["id"],
                "document": case["document"],
                "task_type": case["task_type"],
                "expected_points": case["expected_points"],
                "record": record,
                "history_messages": history_messages,
                "evaluation_context": evaluation_context,
            }

            rag.save_record(eval_record, EVAL_RESULTS_PATH)
            print("评测结果已保存到：", EVAL_RESULTS_PATH)

if __name__ == "__main__":
    main()
