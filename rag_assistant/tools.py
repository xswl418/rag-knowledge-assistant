import json
from rag_assistant.api_client import create_api_client
from rag_assistant.rag import LLM_MODEL_NAME, retrieve_top_k
from rag_assistant.document_task import (
    generate_batched_document_summary,
    generate_batched_document_extraction,
    validate_extraction_fields,
)

available_tools = {
    "search_knowledge": retrieve_top_k,
    "summarize_document": generate_batched_document_summary,
    "extract_document": generate_batched_document_extraction,
}

def validate_tool_request(tool_name, tool_arguments):
    if tool_name not in available_tools:
        return False, f"找不到这个工具：{tool_name}"

    if not isinstance(tool_arguments, dict):
        return False, "工具参数必须是字典"

    if tool_name == "summarize_document":
        if tool_arguments:
            return False, "全文概括工具不接受额外参数"

        return True, None

    if tool_name == "extract_document":
        if set(tool_arguments) != {"fields"}:
            return False, "信息提取工具必须且仅提供fields参数。"

        return validate_extraction_fields(tool_arguments["fields"])

    search_question = tool_arguments.get("question")

    if not isinstance(search_question, str):
        return False, "question参数缺失，或者不是字符串"

    if search_question.strip() == "":
        return False, "检索问题不能为空"

    return True, None

tool_definitions = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "搜索本地知识库，查找与问题相关的参考资料。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "用于检索资料的问题"
                    }
                },
                "required": ["question"],
                "additionalProperties": False
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_document",
            "description": (
                "概括当前文档的全部可提取文字。"
                "用于用户要求总结整份文档、概括全文或整理全文主要内容的任务。"
                "不用于只查找某个具体事实。"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_document",
            "description": (
                "从当前文档中提取用户指定的字段，并保留原文依据。"
                "用于按字段整理或提取信息，不用于概括全文。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fields": {
                        "type": "array",
                        "description": "用户要求提取的字段名称列表",
                        "items": {
                            "type": "string",
                            "minLength": 1,
                        },
                        "minItems": 1,
                        "uniqueItems": True,
                    },
                },
                "required": ["fields"],
                "additionalProperties": False,
            },
        },
    },
]

if __name__ == "__main__":
    client = create_api_client()

    response = client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[
            {"role": "user", "content": "请查询知识库：RAG是什么？"}
        ],
        tools=tool_definitions,
        tool_choice="required"
    )

    message = response.choices[0].message
    print("回答文本：", message.content)
    print("工具调用请求：", message.tool_calls)

    if message.tool_calls:
        tool_call = message.tool_calls[0]

        tool_name = tool_call.function.name
        arguments_text = tool_call.function.arguments

        tool_arguments = json.loads(arguments_text)

        print("工具名称：", tool_name)
        print("转换前的类型：", type(arguments_text))
        print("转换后的类型：", type(tool_arguments))

        is_valid, error_message = validate_tool_request(
            tool_name, tool_arguments
        )

        if is_valid:
            search_tool = available_tools[tool_name]
            print("已找到对应函数：", search_tool.__name__)
            print("检索问题：", tool_arguments["question"])

        else:
            print(error_message)
    else:
        print("本次没有返回工具调用请求")
