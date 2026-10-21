from rag_assistant import rag
from pathlib import Path
from scripts.analyze_logs import load_records

BASE_DIR = Path(__file__).resolve().parent.parent
REVIEWS_PATH = BASE_DIR / "logs" / "eval_reviews.jsonl"
RESULTS_PATH = BASE_DIR / "logs" / "eval_results.jsonl"

def is_completed_run(record):
    status = record["status"]
    answer = record["answer"]

    if status != "answered":
        return False
    elif not isinstance(answer, str) or not answer.strip():
        return False
    else:
        return True

def has_final_review(evaluation, reviews):
    for saved in reviews:
        if (
            saved["evaluation"] == evaluation
            and saved["review"]["verdict"] in ["pass", "fail"]
        ):
            return True

    return False

def main():
    records = load_records(RESULTS_PATH)
    print("评测运行记录数：", len(records))

    if not records:
        print("暂无评测记录。")
        return

    case_ids = set()

    for evaluation in records:
        case_ids.add(evaluation["case_id"])

    print("历史运行涉及的用例编号数:", len(case_ids))

    if REVIEWS_PATH.exists():
        reviews = load_records(REVIEWS_PATH)
    else:
        reviews = []

    reviewed_count = 0
    incomplete_count = 0
    to_review_count = 0

    for latest in records:
        if has_final_review(latest, reviews):
            reviewed_count += 1
            print("已复核，跳过：", latest["case_id"])
            continue

        agent_record = latest["record"]

        if not is_completed_run(agent_record):
            incomplete_count += 1
            print("运行未完成，跳过:", latest["case_id"], agent_record["status"])
            continue

        to_review_count += 1

        print("用例编号：", latest["case_id"])
        print("问题：", agent_record["question"])
        print("预期要点：", latest["expected_points"])
        print("运行状态：", agent_record["status"])
        print("实际回答：", agent_record["answer"])

        while True:
            verdict = input("内容判定 （pass/fail/pending）:").strip()

            if verdict in ["pass", "fail", "pending"]:
                break

            print("输入无效，请重新输入pass、fail或pending.")

        reason = input("判定依据：").strip()

        review = {
            "verdict": verdict,
            "reason": reason,
        }

        print("人工复核结果：", review)

        review_record = {
            "evaluation": latest,
            "review": review,
        }

        rag.save_record(review_record, REVIEWS_PATH)
        reviews.append(review_record)
        print("人工复核已保存到：", REVIEWS_PATH)

    print("已复核，跳过次数：", reviewed_count)
    print("运行未完成次数：", incomplete_count)
    print("本次进入内容复核次数：", to_review_count)

if __name__ == "__main__":
    main()
