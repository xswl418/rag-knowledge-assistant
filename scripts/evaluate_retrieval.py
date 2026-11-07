import json
import jieba

from rank_bm25 import BM25Okapi
from pathlib import Path
from sentence_transformers import SentenceTransformer
from rag_assistant.rag import (
    load_document_chunks,
    create_embeddings,
    retrieve_top_k,
    BASE_DIR,
    MAX_CHUNK_SIZE,
    CHUNK_OVERLAP,
    MINIMUM_SCORE,
)

DOCUMENT_PATH = BASE_DIR / "data" / "samples" / "library_guide.txt"
EVAL_PATH = BASE_DIR / "data" / "eval_questions.jsonl"
MODEL_PATH = BASE_DIR / "models" / "bge-small-zh-v1.5"

def load_eval_questions(path):
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions

def evaluate(chunks, vectors, model, questions, bm25_index, method):
    hits = 0
    reciprocal_rank_sum = 0
    total = len(questions)

    for item in questions:
        question = item["question"]
        gold_ids = item["gold_chunk_ids"]

        if method == "vector":
            results = retrieve_top_k(question, chunks, vectors, model, top_k=3, minimum_score=MINIMUM_SCORE)
        elif method == "bm25":
            results = retrieve_bm25(question, chunks, bm25_index, top_k=3)
        elif method == "rrf":
            results = retrieve_rrf(question, chunks, vectors, model, bm25_index, top_k=3)

        top3_ids = [r["chunk"]["chunk_id"] for r in results]

        if any(gid in top3_ids for gid in gold_ids):
            hits += 1
            ranks = [top3_ids.index(gid) + 1 for gid in gold_ids if gid in top3_ids]
            rank = min(ranks)
            reciprocal_rank_sum += 1 / rank
        else:
            print(f"未命中： {question} (gold_id={gold_ids}, top3={top3_ids})")

    return {
        "method": method,
        "recall@3": hits / total,
        "mrr": reciprocal_rank_sum / total,
        "total": total,
        "hits": hits,
    }

def build_bm25_index(chunks):
    """
    对文档块分词，建立 BM25 索引

    输入：chunks 列表
    输出：BM25Okapi 对象
    """
    tokenized_corpus = [
        jieba.lcut(chunk["content"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

def retrieve_bm25(question, chunks, bm25_index, top_k):
    """
    用 BM25 检索 top_k 个文档块

    输入：
        question: 问题字符串
        chunks: 文档块列表
        bm25_index: BM25Okapi 对象
        top_k: 取前几个
    输出：
        [{"chunk": {...}, "score": float}, ...]  ← 和 retrieve_top_k 格式一致
    """
    scored_chunks = []

    query_tokens = jieba.lcut(question)
    scores = bm25_index.get_scores(query_tokens)

    for index in range(len(chunks)):
        item = {
            "chunk": chunks[index],
            "score": scores[index],
        }
        scored_chunks.append(item)

    scored_chunks.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return scored_chunks[:top_k]

def retrieve_rrf(question, chunks, vectors, model, bm25_index, top_k, k=60):
    """
    RRF 融合检索

    输入：
        question: 问题字符串
        chunks: 文档块列表
        vectors: 文档块向量
        model: 向量模型
        bm25_index: BM25 索引
        top_k: 取前几个
        k: RRF 平衡参数（默认60）
    输出：
        [{"chunk": {...}, "score": float}, ...]
    """
    vectors_results = retrieve_top_k(question, chunks, vectors, model, top_k=10, minimum_score=0)

    bm25_results = retrieve_bm25(question, chunks, bm25_index, top_k=10)

    rrf_scores = {}
    for rank, item in enumerate(vectors_results):
        chunk_id = item["chunk"]["chunk_id"]
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (rank + 1 + k)

    for rank, item in enumerate(bm25_results):
        chunk_id = item["chunk"]["chunk_id"]
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (rank + 1 + k)

    chunk_map = {chunk["chunk_id"]: chunk for chunk in chunks}
    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

    return [
        {"chunk": chunk_map[cid], "score": rrf_scores[cid]}
        for cid in sorted_ids[:top_k]
    ]

def main():
    print("加载文档...")
    chunks = load_document_chunks(DOCUMENT_PATH, MAX_CHUNK_SIZE, CHUNK_OVERLAP)

    print("加载向量模型...")
    model = SentenceTransformer(str(MODEL_PATH), local_files_only=True)

    print("计算文档向量...")
    vectors = create_embeddings(chunks, model)

    print("建立索引...")
    bm25_index = build_bm25_index(chunks)

    print("加载评测问题...")
    questions = load_eval_questions(EVAL_PATH)
    print(f"共 {len(questions)} 道题")


    # 三种方法各跑一次
    for method in ["vector", "bm25", "rrf"]:
        print(f"\n开始评测 {method}...")
        results = evaluate(chunks, vectors, model, questions, bm25_index, method)

        print(f"\n===== 评测结果（{method}）=====")
        print(f"总题数：{results['total']}")
        print(f"命中题数：{results['hits']}")
        print(f"Recall@3：{results['recall@3']:.3f}")
        print(f"MRR：{results['mrr']:.3f}")

        # 保存结果
        result_path = BASE_DIR / "data" / "eval_results.jsonl"
        with open(result_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "method": method,
                "recall@3": results["recall@3"],
                "mrr": results["mrr"],
                "hits": results["hits"],
                "total": results["total"],
            }, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
