import json
from rag_assistant.streaming import merge_tool_call_delta


if __name__ == "__main__":
    tool_calls = {}

    deltas = [
        {
            "index": 0,
            "id": "call_demo",
            "function": {
                "name": "search_knowledge",
                "arguments": '{"question":',
            },
        },
        {
            "index": 1,
            "id": "call_summary",
            "function": {
                "name": "summarize_document",
                "arguments": "{",
            },
        },
        {
            "index": 0,
            "function": {
                "arguments": '"借阅期限是多少天？"}',
            },
        },
        {
            "index": 1,
            "function": {
                "arguments": "}",
            },
        },
    ]

    for delta in deltas:
        merge_tool_call_delta(tool_calls, delta)
        index = delta["index"]
        print(f"工具{index}当前参数：", tool_calls[index]["function"]["arguments"])

    for index, tool_call in tool_calls.items():
        arguments_text = tool_call["function"]["arguments"]
        tool_arguments = json.loads(arguments_text)

        print("工具编号：", index)
        print("工具名称：", tool_call["function"]["name"])
        print("解析后的参数：", tool_arguments)
