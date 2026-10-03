import time
import json
import re
from pathlib import Path
from docx import Document

from openai import (
    OpenAI,
    APIConnectionError,
    AuthenticationError,
    RateLimitError
)

from sentence_transformers import SentenceTransformer, util

BASE_DIR = Path(__file__).resolve().parent
LOG_PATH = BASE_DIR / "query_logs.jsonl"
FILE_PATH = BASE_DIR / "knowledge_base.txt"

SOURCE_NAME = "knowledge_base.txt"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
LLM_MODEL_NAME = "gpt-5.5"

MAX_CHUNK_SIZE = 200
CHUNK_OVERLAP = 20
TOP_K = 3
MINIMUM_SCORE = 0.55

def save_record(record, file_path):
    line = json.dumps(record, ensure_ascii=False)

    with open(file_path, "a", encoding="utf-8") as file:
        file.write(line + "\n")

def generate_answer(prompt, client, model_name):
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role":"user",
                    "content":prompt
                }
            ]
        )

        return response.choices[0].message.content

    except APIConnectionError:
        print("无法连接到模型接口，请检查网络、代理和Base URL")
        return None

    except AuthenticationError:
        print("API Key 无效，请检查环境变量配置")
        return None

    except RateLimitError:
        print("API调用额度或者频率达到限制")
        return None


def load_document(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def load_docx_paragraph(file_path):
    word_document = Document(file_path)
    paragraphs = []

    for paragraph in word_document.paragraphs:
        text = paragraph.text.strip()

        if text == "":
            continue

        item = {
            "text": text,
            "style": paragraph.style.name,
        }
        paragraphs.append(item)

    return paragraphs

def group_docx_sections(paragraphs):
    sections = []
    current_texts = []
    has_heading = False

    for item in paragraphs:
        if item["style"] == "Heading 1":
            if has_heading:
                sections.append("\n".join(current_texts))
                current_texts = []

            has_heading = True

        current_texts.append(item["text"])

    if not has_heading:
        return current_texts

    last_section = "\n".join(current_texts)
    if last_section != "":
        sections.append(last_section)

    return sections

def load_document_sections(file_path):
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        text = load_document(file_path)
        return split_sections(text)

    elif suffix == ".docx":
        paragraphs = load_docx_paragraph(file_path)
        return group_docx_sections(paragraphs)

    else:
        raise ValueError("目前只支持TXT和DOCX文件")

def create_embeddings(chunks, model):
    chunk_texts = []

    for chunk in chunks:
        chunk_texts.append(chunk["content"])

    return model.encode(chunk_texts)

def is_section_heading(line):
    pattern = r"^[一二三四五六七八九十百]+、"
    return re.match(pattern, line.strip()) is not None

def split_sections(text):
    sections = []
    current_lines = []
    has_heading = False

    for line in text.splitlines():
        if is_section_heading(line):
            if has_heading:
                sections.append("\n".join(current_lines).strip())
                current_lines = []

            has_heading = True

        current_lines.append(line)

    if not has_heading:
        return text.split("\n\n")

    last_section = "\n".join(current_lines).strip()
    if last_section != "":
        sections.append(last_section)

    return sections

def split_sentences(text):
    sentences = []
    current_sentence = ""

    for character in text:
        current_sentence += character

        if character in "。！？":
            sentences.append(current_sentence.strip())
            current_sentence = ""

    if current_sentence.strip() != "":
        sentences.append(current_sentence.strip())

    return sentences

def build_sentence_chunks(text, max_chunk_size, overlap=0):
    if max_chunk_size <= 0:
        raise ValueError("知识块长度必须大于0")

    if overlap < 0 or overlap >= max_chunk_size:
        raise ValueError("重叠长度必须大于等于0，并且小于知识块长度")

    sentences = split_sentences(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(sentence) > max_chunk_size:
            if current_chunk != "":
                chunks.append(current_chunk)
                current_chunk = ""

            start = 0

            while start < len(sentence):
                end = start + max_chunk_size
                chunks.append(sentence[start:end])

                if end >= len(sentence):
                    break

                start = end - overlap

            continue

        if len(current_chunk) +len(sentence) > max_chunk_size:
            chunks.append(current_chunk)

            last_sentence = split_sentences(current_chunk)[-1]

            if (
                len(last_sentence) <= overlap
                and len(last_sentence) + len(sentence) <= max_chunk_size
            ):
                current_chunk = last_sentence
            else:
                current_chunk = ""

        current_chunk += sentence

    if current_chunk != "":
        chunks.append(current_chunk)

    return chunks

def split_document(text, source, max_chunk_size, overlap):
    paragraphs = split_sections(text)

    return build_knowledge_chunks(
        paragraphs,
        source,
        max_chunk_size,
        overlap
    )

def build_knowledge_chunks(paragraphs, source, max_chunk_size, overlap):
    if overlap >= max_chunk_size:
        print("重叠长度必须小于知识块长度")
        return []

    knowledge_chunks = []
    paragraph_id = 0

    for paragraph in paragraphs:
        clean_paragraph = paragraph.strip()

        if clean_paragraph != "":
            paragraph_id += 1

            if len(clean_paragraph) <= max_chunk_size:
                chunk = {
                    "chunk_id": len(knowledge_chunks) + 1,
                    "paragraph_id": paragraph_id,
                    "source": source,
                    "content": clean_paragraph
                }

                knowledge_chunks.append(chunk)

            else:
                chunk_texts = build_sentence_chunks(
                    clean_paragraph,
                    max_chunk_size,
                    overlap
                )

                for chunk_text in chunk_texts:
                    chunk = {
                        "chunk_id": len(knowledge_chunks) +1,
                        "paragraph_id": paragraph_id,
                        "source": source,
                        "content": chunk_text
                    }

                    knowledge_chunks.append(chunk)

    return knowledge_chunks


def retrieve_top_k(question, chunks, vectors, model, top_k, minimum_score):
    question_vector = model.encode(question)
    scores = util.cos_sim(question_vector, vectors)[0]

    scored_chunks = []

    for index in range(len(chunks)):
        item = {
            "chunk": chunks[index],
            "score": scores[index].item()
        }
        scored_chunks.append(item)

    scored_chunks.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    results = []

    for item in scored_chunks[:top_k]:
        if item["score"]>=minimum_score:
            results.append(item)

    return results

def build_prompt(question, retrieved_results):
    reference_parts = []

    for result in retrieved_results:
        chunk = result["chunk"]

        part = (
            f"来源：{chunk['source']}，第{chunk['paragraph_id']}段\n"
            f"内容：{chunk['content']}"
        )

        reference_parts.append(part)

    reference_text = "\n\n".join(reference_parts)

    prompt = f"""
你是一个知识库问答助手。

请只根据下面的参考资料回答用户问题。
如果参考资料不足，请明确回答“资料不足”，不得编造。
回答结束后，请列出使用到的资料来源和段落编号。

参考资料：
{reference_text}

用户问题：
{question}
"""

    return prompt

def main():
    document = load_document(FILE_PATH)

    knowledge_chunks = split_document(
        document,
        SOURCE_NAME,
        max_chunk_size=MAX_CHUNK_SIZE,
        overlap=CHUNK_OVERLAP
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        local_files_only=True
    )

    chunk_vectors = create_embeddings(
        knowledge_chunks,
        model
    )

    client = OpenAI()

    available_tools = {
        "search_knowledge": retrieve_top_k
    }

    while True:
        question = input("请输入问题：").strip()

        if question == "exit":
            break

        if question == "":
            print("问题不能为空")
            continue
        question_start = time.perf_counter()

        generation_seconds = None

        print("本次收到的问题：", question)

        retrieval_start = time.perf_counter()

        tool_request = {
            "name": "search_knowledge",
            "arguments": {
                "question": question
            }
        }

        tool_name = tool_request["name"]
        tool_arguments = tool_request["arguments"]

        search_tool = available_tools[tool_name]

        results = search_tool(
            **tool_arguments,
            chunks=knowledge_chunks,
            vectors=chunk_vectors,
            model=model,
            top_k=TOP_K,
            minimum_score=MINIMUM_SCORE
        )

        retrieval_seconds = time.perf_counter() - retrieval_start
        print(f"检索耗时：{retrieval_seconds:.3f}秒")

        if len(results) > 0:
            prompt = build_prompt(question, results)

            print("\n发送给大模型的完整指令：")
            print(prompt)

            generation_start = time.perf_counter()

            answer = generate_answer(
                prompt,
                client,
                LLM_MODEL_NAME
            )

            generation_seconds = time.perf_counter() - generation_start
            print(f"生成接口调用耗时：{generation_seconds:.3f}秒")
            if answer is not None and answer.strip() != "":
                status = "answered"
                print("\n大模型回答：")
                print(answer)

            else:
                status = "no_answer"
                print("本次未获得有效回答")

        else:
            status = "no_results"
            print("没有找到足够相关的内容")

        total_seconds = time.perf_counter() - question_start
        print(f"本次问题总处理耗时：{total_seconds:.3f}秒")

        record = {
            "question": question,
            "retrieved_count": len(results),
            "retrieval_seconds": retrieval_seconds,
            "generation_seconds": generation_seconds,
            "total_seconds": total_seconds,
            "status": status
        }

        print(record)

        save_record(record, LOG_PATH)

if __name__ == "__main__":
    main()
