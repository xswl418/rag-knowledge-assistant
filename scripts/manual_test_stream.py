from rag_assistant.api_client import create_api_client
from rag_assistant.rag import LLM_MODEL_NAME
from openai import APIError
from httpx import  TransportError
from rag_assistant.streaming import collect_stream_response
from rag_assistant.tools import (tool_definitions, validate_tool_request)


import json

def print_text_piece(text):
    print(text, end="", flush=True)

if __name__ == "__main__":
    finish_reason = None
    answer = ""
    stream_error = None
    tool_calls = []

    with create_api_client() as client:
        try:
            with client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=[
                    {"role": "user", "content": "请检索文档：青禾书屋的借阅期限是多少天？"}
                ],
                stream=True,
                tools=tool_definitions,
                tool_choice="required",
            ) as stream:
                result = collect_stream_response(
                    stream,
                    on_text=print_text_piece,
                )
                answer = result["content"].strip()
                tool_calls = result["tool_calls"]
                finish_reason = result["finish_reason"]

        except (APIError, TransportError) as error:
            stream_error = type(error).__name__

    print()
    print("结束原因：", finish_reason)

    if stream_error is not None:
        print(f"请求或接收失败：{stream_error}，本次结果不作为完整回答。")
    elif finish_reason == "tool_calls" and tool_calls:
        for index, tool_call in enumerate(tool_calls):
            tool_name = tool_call["function"]["name"]
            arguments_text = tool_call["function"]["arguments"]

            print("工具编号：", index)
            print("工具名称：", tool_call["function"]["name"])

            try:
                tool_arguments = json.loads(arguments_text)
            except json.JSONDecodeError :
                print("参数不是合法的JSON，本条请求不执行。")
                continue

            is_valid, error_message = validate_tool_request(
                tool_name,
                tool_arguments,
            )

            if not is_valid:
                print("工具请求校验失败：", error_message)
                continue

            print("工具请求校验通过，尚未执行。")
            print("参数字典：", tool_arguments)

    elif finish_reason != "stop" or tool_calls:
        print("本次未正常完成，已收到的文字不作为完整回答。")
    elif not answer:
        print("本次没有收到有效的回答文字。")
    else:
        print("本次回答接收成功。")
        print("完整回答：", answer)
