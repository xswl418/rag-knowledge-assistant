"""固定提供相关但不足的资料，人工评测 Agent 的真实回答。"""

from unittest.mock import patch

from openai import OpenAI

import rag
import agent


QUESTION = "RAG 的检索相似度阈值应该设置为多少？"


def fixed_search(question, chunks, vectors, model, top_k, minimum_score):
    results = []

    for chunk in chunks:
        if chunk["paragraph_id"] == 1:
            # None 表示本次跳过相似度计算，不伪造检索分数。
            results.append({"chunk": chunk, "score": None})

    print("[受控评测] 固定返回第 1 段，没有执行向量检索。")
    return results


def main():
    document = rag.load_document(rag.FILE_PATH)
    chunks = rag.split_document(
        document,
        rag.SOURCE_NAME,
        max_chunk_size=rag.MAX_CHUNK_SIZE,
        overlap=rag.CHUNK_OVERLAP
    )

    if not any(chunk["paragraph_id"] == 1 for chunk in chunks):
        print("知识库没有第 1 段，无法执行本次评测。")
        return

    print("评测问题：", QUESTION)
    print("模型配置名：", rag.LLM_MODEL_NAME)
    print("固定资料：", rag.SOURCE_NAME, "第 1 段")
    print("相似度为 None：本次没有计算相似度。")
    print("通过标准：说明资料不足，不提供无依据的阈值或设置建议。")

    with OpenAI(timeout=30.0, max_retries=0) as client:
        with patch.dict(agent.available_tools, {"search_knowledge": fixed_search}):
            record = agent.run_agent(
                question=QUESTION,
                client=client,
                knowledge_chunks=chunks,
                chunk_vectors=None,
                embedding_model=None
            )

    print("评测运行记录：", record)

    if record["tool_call_count"] == 0:
        print("未覆盖目标场景：模型没有执行固定资料工具。")
    elif record["status"] != "answered":
        print("未获得固定资料后的有效回答，本次不能判定回答质量通过。")
    else:
        print("已获得固定资料后的回复；请核对内容，answered 不代表评测通过。")

    return record


if __name__ == "__main__":
    main()
