from agent import load_user_document
import rag


if __name__ == "__main__":
    # print(rag.is_section_heading("一、开放与入馆"))
    # print(rag.is_section_heading("十二、其他说明"))
    # print(rag.is_section_heading("青禾书屋使用指南"))
    # print(rag.is_section_heading("每位读者最多可以同时借阅三本书。"))
    #
    # test_chunks = rag.build_sentence_chunks(
    #     "今天开放。明天闭馆。请带证件。",
    #     max_chunk_size=10,
    #     overlap=5
    # )
    #
    # for chunk in test_chunks:
    #     print(len(chunk), chunk)

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

    # sections = rag.split_sections(document)
    #
    # print("分组数量：", len(sections))
    #
    # for section in sections:
    #     print("\n这一组的内容：")
    #     print(section)