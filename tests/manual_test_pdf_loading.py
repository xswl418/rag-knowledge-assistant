from pathlib import Path
from typing import overload

import rag


if __name__ == "__main__":
    file_path_text = input("请输入 PDF 文件的完整路径：").strip().strip('"')
    file_path = Path(file_path_text)

    knowledge_chunks = rag.load_document_chunks(
        file_path,
        max_chunk_size=rag.MAX_CHUNK_SIZE,
        overlap=rag.CHUNK_OVERLAP
    )

    for chunk in knowledge_chunks:
        print("\n知识块编号：", chunk["chunk_id"])
        print("来源页码：", chunk["page_number"])
        print("内容：", chunk["content"])