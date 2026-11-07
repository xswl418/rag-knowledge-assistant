from rag_assistant import rag
import json
import time
import logging

from pypdf.errors import PdfReadError
from zipfile import BadZipFile
from docx.opc.exceptions import PackageNotFoundError
from pathlib import Path
from openai import (
    APITimeoutError,
    APIConnectionError,
    AuthenticationError,
    RateLimitError,
    APIStatusError,
    APIError,
)
from rag_assistant.api_client import create_api_client
from rag_assistant.tools import (
    available_tools,
    validate_tool_request,
    tool_definitions
)
from rag_assistant.document_task import (
    build_document_context,
    MAX_DOCUMENT_BATCHES,
)
from rag_assistant.streaming import collect_stream_response
from openai.types.chat import ChatCompletionMessage
from httpx import TransportError

logger = logging.getLogger(__name__)

AGENT_LOG_PATH = rag.BASE_DIR / "logs" / "agent_logs.jsonl"

def build_search_content(results, remaining_search_calls):
    if not results:
        return {
            "results": [],
            "message": "本次检索为空。请围绕用户原始问题换一种问法，再调用 search_knowledge,不要重复原查询。",
            "remaining_search_calls": remaining_search_calls,
        }
    elif results[0].get("low_confidence"):
        return {
            "results": results,
            "message": "结果相似度低于阈值，可能不相关；如果不能回答问题且还有检索次数，换个问法再调用search_knowledge，否则说明资料不足。",
            "remaining_search_calls": remaining_search_calls,
        }
    else:
        return results


def call_llm_with_retry(client, messages, tool_definitions, tool_choice, on_text=None, max_retries=3):
    """
    调用大模型 API，支持重试。

    参数：
        client: OpenAI 客户端
        messages: 对话历史
        tool_definitions: 工具定义
        tool_choice: 工具选择策略
        on_text: 文本流式回调函数
        max_retries: 最大重试次数（默认3次）

    返回：
        (success, result, error_info)
        - success: bool，是否成功
        - result: 成功时返回 stream_result，失败时为 None
        - error_info: 失败时返回错误信息字典，成功时为 None
                     {"status": "api_error", "message": "..."}
    """
    for retry_count in range(max_retries):
        try:
            with client.chat.completions.create(
                    model=rag.LLM_MODEL_NAME,
                    messages=messages,
                    tools=tool_definitions,
                    tool_choice=tool_choice,
                    stream=True,
            ) as stream:
                stream_result = collect_stream_response(
                    stream,
                    on_text=on_text,
                )
            # 成功：返回结果
            return (True, stream_result, None)

        except (APITimeoutError, APIConnectionError, RateLimitError) as e:
            error_type = type(e).__name__
            logger.warning(f"可重试的错误：%s，等待后重试...")
            if retry_count < max_retries - 1:  # 还有重试机会
                time.sleep(2 ** retry_count)
                continue
            else:  # 重试耗尽
                return (False, None, {
                    "status": "api_error",
                    "message": f"重试{max_retries}次后仍失败：{error_type}"
                })

        except APIStatusError as error:
            if error.status_code >= 500:
                logger.warning(f"服务端错误 %d，等待后重试...")
                if retry_count < max_retries - 1:
                    time.sleep(2 ** retry_count)
                    continue
                else:
                    return (False, None, {
                        "status": "api_error",
                        "message": f"重试{max_retries}次后仍失败：HTTP {error.status_code}"
                    })
            else:  # 4xx 客户端错误，不重试
                return (False, None, {
                    "status": "api_error",
                    "message": f"客户端错误，不可重试：HTTP {error.status_code}"
                })

        except (APIError, TransportError, AuthenticationError) as e:
            error_type = type(e).__name__
            return (False, None, {
                "status": "api_error",
                "message": f"不可重试的错误：{error_type}"
            })

    # 理论上不会到这里，但为了安全
    return (False, None, {
        "status": "api_error",
        "message": "未知错误：重试逻辑异常退出"
    })


def run_agent(
        question,
        client,
        knowledge_chunks,
        chunk_vectors,
        embedding_model,
        history_messages=None,
        get_retrieval_resources=None,
        on_summary_progress=None,
        on_text=None,
        on_extraction_progress=None,
        task_id: int | None = None,
):
    task_start = time.perf_counter()
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个知识库助手。"
                "当前任务围绕用户已上传的文档。"
                "用户未明确切换对象时，具体事实问题默认针对当前文档，先调用 search_knowledge 查找依据。"
                "不要仅因用户没有提供对象名称就要求澄清。"
                "结合历史问答理解当前问题中的指代；无法确定指代时，先请客户澄清。"
                "调用 search_knowledge 时，将追问改写为包含具体主题和需求的完整检索问题。"
                "历史回答只用于理解上下文，不作为文档证据。"
                "对于普通的具体事实、条件或条款问答，调用 search_knowledge，再依据检索资料回答。"
                "用户明确要求按指定字段提取或整理信息时，调用 extract_document，fields 使用用户要求的字段名称。"
                "用户要求提取但未说明字段时，先请用户明确需要哪些字段。"
                "对于总结整份文档、概括全文或整理全文主要内容的请求，调用 summarize_document。"
                "不得用少量检索片段冒充全文概括。"
                "资料不足时明确说明，不得编造；使用资料时注明来源文件；有 page_number 时引用该页码。"
                "没有时引用 paragraph_id 分组编号，不得编造来源位置。"
                "对于简单问候或感谢，可以直接简短回复，不需要搜索知识库。"
            )
        },
    ]

    if history_messages is not None:
        messages.extend(history_messages)

    messages.append({
        "role": "user",
        "content": question
    })
    max_search_calls = 3
    search_call_count = 0

    max_agent_model_calls = max_search_calls + 1
    agent_model_call_count = 0

    max_model_calls = max_agent_model_calls + MAX_DOCUMENT_BATCHES + 1
    model_call_count = 0

    tool_call_count = 0
    model_timings = []
    search_queries = []
    retrieval_steps = []

    status = "unknown"
    final_answer = None
    task_error = None
    summary_input = None
    extraction_fields = []
    extraction_result = None
    extraction_input = None

    while True:
        if (
            agent_model_call_count >= max_agent_model_calls
            or model_call_count >= max_model_calls
        ):
            status = "max_calls"
            print("已达到模型调用次数上限，本次处理结束")
            break

        agent_model_call_count += 1
        model_call_count += 1
        print(f"第{model_call_count} 次调用模型")
        model_start = time.perf_counter()

        tool_choice = "auto"

        if (
            retrieval_steps
            and not retrieval_steps[-1]["results"]
            and search_call_count < max_search_calls
        ):
            tool_choice = {
                "type": "function",
                "function": {"name": "search_knowledge"},
            }
        success, stream_result, error_info = call_llm_with_retry(
            client=client,
            messages=messages,
            tool_definitions=tool_definitions,
            tool_choice=tool_choice,
            on_text=on_text,
        )

        model_timings.append({
            "stage": "agent_model",
            "seconds": time.perf_counter() - model_start,
        })

        if not success:
            status = error_info["status"]
            task_error = error_info["message"]
            print(task_error)
            break



        expected_reason = (
            "tool_calls" if stream_result["tool_calls"] else "stop"
        )

        if stream_result["finish_reason"] != expected_reason:
            status = "no_answer"
            task_error = "模型未正常完成，本次不执行工具或保存为成功回答。"
            print(task_error)
            break

        message = ChatCompletionMessage(
            role="assistant",
            content=stream_result["content"] or None,
            tool_calls=stream_result["tool_calls"] or None,
        )

        if isinstance(tool_choice, dict):
            if (
                len(message.tool_calls or []) != 1
                or message.tool_calls[0].function.name != "search_knowledge"
            ):
                status = "no_answer"
                task_error = "检索重试未返回指定的检索调用，本次处理结束。"
                print(task_error)
                break

        if not message.tool_calls:
            answer = message.content

            if answer is not None and answer.strip() != "":
                status = "answered"
                final_answer = answer.strip()
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

            if tool_name == "summarize_document":
                tool_call_count += 1

                context = build_document_context(knowledge_chunks)

                summary_input = {
                    "char_count": len(context),
                    "chunk_count": len(knowledge_chunks),
                    "sources": [],
                }

                for chunk in knowledge_chunks:
                    summary_input["sources"].append({
                        "source": chunk["source"],
                        "chunk_id": chunk["chunk_id"],
                        "paragraph_id": chunk.get("paragraph_id"),
                        "page_number": chunk.get("page_number"),
                    })

                summary_tool = available_tools[tool_name]
                remaining_model_calls = (
                    max_model_calls - model_call_count
                )

                print("开始生成文档摘要")

                summary, task_error, summary_calls_used = summary_tool(
                    chunks=knowledge_chunks,
                    client=client,
                    model_name=rag.LLM_MODEL_NAME,
                    remaining_model_calls=remaining_model_calls,
                    model_timings=model_timings,
                    summary_details=summary_input,
                    on_progress=on_summary_progress,
                    on_text=on_text,
                )

                model_call_count += summary_calls_used

                if task_error is not None:
                    status = "summary_error"
                    print(task_error)
                else:
                    status = "answered"
                    final_answer =summary
                    print("最终摘要:", summary)

                break

            if tool_name == "extract_document":
                tool_call_count += 1
                extraction_fields = tool_arguments["fields"]

                context = build_document_context(knowledge_chunks)

                extraction_tool = available_tools[tool_name]

                extraction_input = {
                    "char_count": len(context),
                    "chunk_count": len(knowledge_chunks),
                    "sources": [],
                }

                for chunk in knowledge_chunks:
                    extraction_input["sources"].append({
                        "source": chunk["source"],
                        "chunk_id": chunk["chunk_id"],
                        "paragraph_id": chunk.get("paragraph_id"),
                        "page_number": chunk.get("page_number"),
                    })

                remaining_model_calls = max_model_calls - model_call_count
                print("开始逐批提取文档信息")

                extraction_result, task_error, extraction_calls_used = extraction_tool(
                    chunks=knowledge_chunks,
                    fields=extraction_fields,
                    client=client,
                    model_name=rag.LLM_MODEL_NAME,
                    remaining_model_calls=remaining_model_calls,
                    model_timings=model_timings,
                    extraction_details=extraction_input,
                    on_progress=on_extraction_progress,
                )

                model_call_count += extraction_calls_used

                if task_error is not None:
                    status = "extraction_error"
                    print(task_error)
                else:
                    status = "answered"
                    final_answer = json.dumps(
                        extraction_result,
                        ensure_ascii=False,
                        indent=2,
                    )
                    print("最终提取结果：:", final_answer)

                break
            if embedding_model is None or chunk_vectors is None:
                if get_retrieval_resources is None:
                    status = "retrieval_error"
                    task_error = "缺少检索资源，暂时无法检索文档。"
                    break

                try:
                    embedding_model, chunk_vectors = get_retrieval_resources()
                except Exception as error:
                    status = "retrieval_error"
                    task_error = (
                        "检索准备失败，请重试；"
                        "若仍失败，请检查本地模型和运行环境。"
                    )
                    print("检索准备异常类型：", type(error).__name__)
                    break

                if embedding_model is None or chunk_vectors is None:
                    status = "retrieval_error"
                    task_error = "检索资源准备不完整，暂时无法检索文档。"
                    break
            if search_call_count >= max_search_calls:
                status = "max_calls"
                print("已达到检索模型次数上限，本次处理结束。")
                break

            search_tool = available_tools[tool_name]

            tool_call_count += 1
            search_call_count += 1
            search_queries.append(tool_arguments["question"])

            results = search_tool(
                question=tool_arguments["question"],
                chunks=knowledge_chunks,
                vectors=chunk_vectors,
                model=embedding_model,
                top_k=rag.TOP_K,
                minimum_score=rag.MINIMUM_SCORE
            )

            retrieval_steps.append({
                "question": tool_arguments["question"],
                "results": results
            })

            print("检索结果数量：", len(results))

            for result in results:
                chunk = result["chunk"]

                print("来源：", chunk["source"])

                if "page_number" in chunk:
                    print("来源页码：", chunk["page_number"])
                else:
                    print("来源分组：", chunk["paragraph_id"])

                print("内容：", chunk["content"])
                print("相似度：", result["score"])

            if not results:
                if search_call_count >= max_search_calls:
                    status = "no_results"
                    print("已检索三次，仍未找到相关资料，本次处理结束。")
                    break

            tool_result_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(build_search_content(results, remaining_search_calls=max_search_calls-search_call_count),
                ensure_ascii=False)
            }

            messages.append(tool_result_message)

    total_seconds = time.perf_counter() - task_start
    print(f"本次任务耗时：{total_seconds:.3f}秒")

    print("本次结束状态：", status)

    record = {
        "task_id": task_id,
        "question": question,
        "status": status,
        "answer": final_answer,
        "error_message": task_error,
        "search_queries": search_queries,
        "retrieval_steps": retrieval_steps,
        "agent_model_call_count": agent_model_call_count,
        "model_call_count": model_call_count,
        "tool_call_count": tool_call_count,
        "total_seconds": total_seconds,
        "model_timings": model_timings,
        "summary_input": summary_input,
        "extraction_fields": extraction_fields,
        "extraction_result":extraction_result,
        "extraction_input": extraction_input
    }

    return record

def load_user_document():
    while True:
        file_path_text = input("请输入TXT、DOCX或PDF文件的完整路径(按exit退出)：").strip().strip('"')

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

        if file_path.suffix.lower() not in(".txt", ".docx", ".pdf") :
            print("目前只支持TXT、DOCX和PDF文件，请重新输入")
            continue

        try:
            knowledge_chunks = rag.load_document_chunks(
                file_path,
                max_chunk_size=rag.MAX_CHUNK_SIZE,
                overlap=rag.CHUNK_OVERLAP
            )

        except UnicodeDecodeError:
            print("文件无法按UTF-8解码，请选择UTF-8编码的TXT文件")
            continue
        except OSError:
            print("文件读取失败，请检查文件是否存在，以及是否有读取权限")
            continue
        except (PackageNotFoundError, BadZipFile):
            print("无法识别这个 Word 文件，请选择有效的 DOCX 文档")
            continue
        except PdfReadError:
            print("PDF读取失败，请选择有效且无需密码的PDF文件")
            continue

        if not knowledge_chunks:
            print("没有读取到可用文字，请检查文件内容")
            continue

        return file_path, knowledge_chunks

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    file_path, knowledge_chunks = load_user_document()

    print("本次读取的文件：", file_path.name)

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

    client = create_api_client()

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
