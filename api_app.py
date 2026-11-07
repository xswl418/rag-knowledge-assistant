from pathlib import Path
from rag_assistant import rag
from fastapi import FastAPI, HTTPException, Query, Depends
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer
from rag_assistant.api_client import  create_api_client
from pydantic import BaseModel, Field
from rag_assistant.agent import run_agent
from rag_assistant.logger_config import setup_logging

import sqlite3
import logging
import time

setup_logging()

logger = logging.getLogger("rag_assistant.api")

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "api_tasks.db"
DOCUMENT_PATH = BASE_DIR / "data" / "samples" / "library_guide.txt"
EMBEDDING_MODEL_PATH = BASE_DIR / "models" / "bge-small-zh-v1.5"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY,
                question TEXT NOT NULL,
                status TEXT NOT NULL,
                answer TEXT,
                error_message TEXT
            )
        """)
        conn.commit()
    finally:
        conn.close()


def load_knowledge_chunks():
    chunks = rag.load_document_chunks(DOCUMENT_PATH, rag.MAX_CHUNK_SIZE, rag.CHUNK_OVERLAP)
    return chunks



@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("任务表已准备完毕")
    app.state.knowledge_chunks = load_knowledge_chunks()
    logger.info("文档加载完成，文本块数量: %d", len(app.state.knowledge_chunks))
    logger.info("开始加载向量模型")
    app.state.embedding_model = SentenceTransformer(
        str(EMBEDDING_MODEL_PATH),
        local_files_only = True,
    )

    logger.info("开始计算文档块向量")
    app.state.chunk_vectors = rag.create_embeddings(app.state.knowledge_chunks, app.state.embedding_model)
    logger.info("向量准备完成，数量：%d", len(app.state.chunk_vectors))

    app.state.model_client = create_api_client()

    try:
        logger.info("生成模型客户端已准备")
        yield
    finally:
        app.state.model_client.close()
        logger.info("生成模型客户端已关闭")

app = FastAPI(lifespan=lifespan)

@app.middleware("http")
async def log_request_time(request, call_next):
    start_time = time.perf_counter()

    response = await call_next(request)
    elapsed_time = time.perf_counter() - start_time

    logger.info("%.3f秒 %s", elapsed_time, request.url.path)

    return response

@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunk_count": len(app.state.knowledge_chunks),
        "vector_count": len(app.state.chunk_vectors),
    }

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)

@app.post("/ask")
def ask(payload: AskRequest):
    task_id = create_task(payload.question)
    record = run_agent(
        question=payload.question,
        client=app.state.model_client,
        knowledge_chunks=app.state.knowledge_chunks,
        chunk_vectors=app.state.chunk_vectors,
        embedding_model=app.state.embedding_model,
        task_id=task_id,
    )
    try:
        update_task(task_id, record)
    except sqlite3.Error:
        logger.exception("任务结果保存失败")
        raise HTTPException(status_code=500, detail="任务结果保存失败")

    rag.save_record(record, BASE_DIR / "logs" / "agent_logs.jsonl")
    return {
        "question": record["question"],
        "status": record["status"],
        "answer": record["answer"],
        "error_message": record["error_message"],
        "id": task_id,
    }

def create_task(question: str) -> int:
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.execute(
            "INSERT INTO tasks (question, status, answer, error_message) VALUES (?, ?, ?, ?)",
            (question, "processing", None, None),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()

def update_task(task_id: int, record: dict):
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "UPDATE tasks SET status = ?, answer = ?, error_message = ? WHERE id = ?",
            (record["status"], record["answer"], record["error_message"], task_id,),
        )
        conn.commit()
    finally:
        conn.close()

@app.get("/tasks/{task_id}")
def get_task(task_id: int, db = Depends(get_db)):
    cursor = db.execute(
        "SELECT id, question, status, answer, error_message FROM tasks WHERE id = ?",
        (task_id,)
    )
    row = cursor.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "id": row[0],
        "question": row[1],
        "status": row[2],
        "answer": row[3],
        "error_message": row[4],
    }

@app.delete("/tasks/{task_id}")
def delete_task(task_id: int, db = Depends(get_db)):
    cursor = db.execute(
        "DELETE FROM tasks WHERE id = ?",
        (task_id,)
    )
    db.commit()

    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Task not found")

    return {"deleted": task_id}

@app.get("/tasks")
def list_tasks(
        limit: int = Query(default=2, ge=1, le=20),
        status: str | None = None,
        db= Depends(get_db)
):
    if status is None:
        cursor = db.execute(
            "SELECT id, question, status, answer, error_message "
            "FROM tasks "
            "ORDER BY id DESC "
            "LIMIT ?",
            (limit,)
        )
    else:
        cursor = db.execute(
            "SELECT id, question, status, answer, error_message "
            "FROM tasks "
            "WHERE status = ? "
            "ORDER BY id DESC "
            "LIMIT ?",
            (status, limit,)
        )
    rows = cursor.fetchall()

    items = []
    for row in rows:
        item = {
            "id": row[0],
            "question": row[1],
            "status": row[2],
            "answer": row[3],
            "error_message": row[4],
        }
        items.append(item)

    return {"items": items}

@app.get("/task-stats")
def task_status(db = Depends(get_db)):
    cursor = db.execute(
        "SELECT status, COUNT(*)"
        " FROM tasks "
        "GROUP BY status "
        "ORDER BY status ASC "
    )
    rows = cursor.fetchall()

    total = 0
    items = []
    for row in rows:
        total += row[1]
        item = {
            "status": row[0],
            "count": row[1],
        }
        items.append(item)

    return {"total": total, "items": items}

if __name__ == "__main__":
    knowledge_chunks = load_knowledge_chunks()
    print("文档：", DOCUMENT_PATH.name)
    print("文本块数量：", len(knowledge_chunks))
    print("第一个文本块：", knowledge_chunks[0])
