from pathlib import Path
import rag


if __name__ == "__main__":
    file_path_text = input("请输入 TXT或DOCX 文件的完整路径：").strip().strip('"')
    file_path = Path(file_path_text)

    sections = rag.load_document_sections(file_path)

    chunks = rag.build_knowledge_chunks(
        sections,
        file_path.name,
        max_chunk_size=rag.MAX_CHUNK_SIZE,
        overlap=rag.CHUNK_OVERLAP,
    )

    print("知识块数量：", len(chunks))

    for chunk in chunks:
        print(chunk)