from unittest.mock import Mock
from manual_test_tool_execution import run_agent


if __name__ == "__main__":
    client = Mock()

    message = Mock(content=None, tool_calls=None)
    choice = Mock(message=message)
    client.chat.completions.create.return_value.choices = [choice]

    first_record = run_agent(
        question="第一个问题",
        client=client,
        knowledge_chunks=[],
        chunk_vectors=None,
        embedding_model=None
    )

    message.content = "这是一条模拟回答"

    second_record = run_agent(
        question="第二个问题",
        client=client,
        knowledge_chunks=[],
        chunk_vectors=None,
        embedding_model=None
    )

    assert first_record["status"] == "no_answer"
    assert second_record["status"] == "answered"

    assert first_record["model_call_count"] == 1
    assert second_record["model_call_count"] == 1

    assert first_record["question"] == "第一个问题"
    assert second_record["question"] == "第二个问题"

    assert first_record["tool_call_count"] == 0
    assert second_record["tool_call_count"] == 0

    print("测试通过：两个问题的状态和调用次数相互独立")