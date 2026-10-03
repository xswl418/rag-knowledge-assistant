import rag
import json
import time

from zipfile import BadZipFile
from docx.opc.exceptions import PackageNotFoundError
from pathlib import Path
from openai import (
    OpenAI,
    APITimeoutError,
    APIConnectionError,
    AuthenticationError,
    RateLimitError,
    APIStatusError
)
from sentence_transformers import SentenceTransformer
from manual_test_tool_call import (
    available_tools,
    validate_tool_request,
    tool_definitions
)

AGENT_LOG_PATH = rag.BASE_DIR / "agent_logs.jsonl"

def run_agent(
        question,
        client,
        knowledge_chunks,
        chunk_vectors,
        embedding_model
):
    task_start = time.perf_counter()
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个知识库助手。"
                "对于知识性问题，先搜索知识库，再依据工具返回的资料回答。"
                "资料不足时明确说明，不得编造；使用资料时注明来源文件和段落编号。"
                "对于简单问候或感谢，可以直接简短回复，不需要搜索知识库。"
            )
        },
        {"role": "user", "content": question}
    ]

    max_model_calls = 3
    model_call_count = 0
    tool_call_count = 0

    status = "unknown"

    while True:
        if model_call_count >= max_model_calls:
            status = "max_calls"
            print("已达到模型调用次数上限，本次处理结束")
            break

        model_call_count += 1
        print(f"第{model_call_count} 次调用模型")

        try:
            response = client.chat.completions.create(
                model=rag.LLM_MODEL_NAME,
                messages=messages,
                tools=tool_definitions,
                tool_choice="auto"
            )
        except APITimeoutError:
            status = "api_error"
            print("模型接口超时，请稍后重试")
            break
        except APIConnectionError:
            status = "api_error"
            print("无法连接模型接口，请检查网络、代理和接口地址")
            break
        except AuthenticationError:
            status = "api_error"
            print("模型接口鉴权失败，请检查API密钥及其对应的接口网址")
            break
        except RateLimitError:
            status = "api_error"
            print("模型接口触发调用限制，请检查请求频率、账户额度或服务商限制")
            break
        except APIStatusError as error:
            status = "api_error"
            print(f"模型接口返回错误，HTTP状态码：{error.status_code}")
            break

        message = response.choices[0].message

        if not message.tool_calls:
            answer = message.content

            if answer is not None and answer.strip() != "":
                status = "answered"
                print("最终回答：", answer.strip())
            else:
                status = "no_answer"
                print("模型未返回有效回答")
            break

        elif len(message.tool_calls) !=1:
            status = "unsupported_tool_calls"
            print("本次示例暂不处理多条工具调用请求")
            break

        else:
            tool_call = message.tool_calls[0]
            tool_name = tool_call.function.name

            messages.append(message.model_dump(exclude_none=True))

            try:
                tool_arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                error_message = "工具参数不是合法的JSON，请检查格式后重新调用"
                print(error_message)

                tool_error_message={
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(
                        {"error": error_message},
                        ensure_ascii=False
                    )
                }

                messages.append(tool_error_message)
                continue

            print("模型请求的工具：", tool_name)
            print("模型给出的参数：", tool_arguments)

            is_valid, error_message = validate_tool_request(tool_name, tool_arguments)

            if not is_valid:
                print(error_message)

                tool_error_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(
                        {"error": error_message},
                        ensure_ascii=False
                    )
                }

                messages.append(tool_error_message)
                continue

            search_tool = available_tools[tool_name]

            tool_call_count += 1

            results = search_tool(
                question=tool_arguments["question"],
                chunks=knowledge_chunks,
                vectors=chunk_vectors,
                model=embedding_model,
                top_k=rag.TOP_K,
                minimum_score=rag.MINIMUM_SCORE
            )

            print("检索结果数量：", len(results))

            for result in results:
                print("来源：", result["chunk"]["source"])
                print("内容：", result["chunk"]["content"])
                print("相似度：", result["score"])

            if not results:
                status = "no_results"
                print("知识库中未找到相关资料")
                break

            else:
                tool_result_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(results, ensure_ascii=False)
                }

                print("准备回传的工具消息：", tool_result_message)

                messages.append(tool_result_message)

    total_seconds = time.perf_counter() - task_start
    print(f"本次任务耗时：{total_seconds:.3f}秒")

    print("本次结束状态：", status)

    record = {
        "question": question,
        "status": status,
        "model_call_count": model_call_count,
        "tool_call_count": tool_call_count,
        "total_seconds": total_seconds,
    }

    return record

def load_user_document():
    while True:
        file_path_text = input("请输入TXT或DOCX文件的完整路径(按exit退出)：").strip().strip('"')

        if file_path_text == "exit":
            print("程序已退出")
            raise SystemExit(0)

        if file_path_text == "":
            print("文件路径不能为空，请重新输入")
            continue

        file_path = Path(file_path_text)

        if not file_path.is_file():
            print("该路径不是现有文件，请检查后重新输入")
            continue

        if file_path.suffix.lower() not in(".txt", ".docx") :
            print("目前只支持TXT和DOCX文件，请重新输入")
            continue

        try:
            sections = rag.load_document_sections(file_path)
        except UnicodeDecodeError:
            print("文件无法按UTF-8解码，请选择UTF-8编码的TXT文件")
            continue
        except OSError:
            print("文件读取失败，请检查文件是否存在，以及是否有读取权限")
            continue
        except (PackageNotFoundError, BadZipFile):
            print("无法识别这个 Word 文件，请选择有效的 DOCX 文档")
            continue

        if "".join(sections).strip() == "":
            print("没有读取到有效正文，请选择其他文件")
            continue

        return file_path, sections

if __name__ == "__main__":
    file_path, sections = load_user_document()

    print("本次读取的文件：", file_path.name)

    knowledge_chunks = rag.build_knowledge_chunks(
        sections,
        file_path.name,
        max_chunk_size=rag.MAX_CHUNK_SIZE,
        overlap=rag.CHUNK_OVERLAP
    )

    print("知识块数量：", len(knowledge_chunks))

    for chunk in knowledge_chunks:
        print("内容：", chunk["content"])

    embedding_model = SentenceTransformer(
        rag.EMBEDDING_MODEL_NAME,
        local_files_only=True
    )

    chunk_vectors = rag.create_embeddings(
        knowledge_chunks,
        embedding_model
    )

    print("向量形状：", chunk_vectors.shape)

    client = OpenAI(timeout=30.0, max_retries=0)

    while True:
        question = input("请输入问题：").strip()

        if question == "exit":
            print("程序已退出")
            break

        if question == "":
            print("问题不能为空，请重新输入")
            continue

        record = run_agent(
            question,
            client,
            knowledge_chunks,
            chunk_vectors,
            embedding_model
        )

        record["source"] = file_path.name
        rag.save_record(record, AGENT_LOG_PATH)
        print("本次任务日志已保存")
