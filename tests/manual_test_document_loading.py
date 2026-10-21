from rag_assistant.agent import load_user_document


if __name__ == "__main__":
    file_path, chunks = load_user_document()

    print("文件名：", file_path.name)

    print("知识块数量：", len(chunks))

    for chunk in chunks:
        print("\n知识块编号：", chunk["chunk_id"])
        print("所属段落编号:", chunk["paragraph_id"])
        print("字符数：", len(chunk["content"]))
        print("内容：", chunk["content"])

        if "page_number" in chunk:
            print("来源页码：", chunk["page_number"])
