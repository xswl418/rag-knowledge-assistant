import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = BASE_DIR / "logs" / "query_logs.jsonl"


def load_records(file_path):
    records = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line == "":
                continue

            record = json.loads(line)
            records.append(record)

    return records


def main():
    records = load_records(LOG_PATH)
    print("日志记录条数：", len(records))

    generation_count = 0
    generation_total_seconds = 0.0
    total_processing_seconds = 0.0

    status_counts = {
        "answered": 0,
        "no_answer": 0,
        "no_results": 0,
        "unknown": 0
    }

    for record in records:
        seconds = record["generation_seconds"]
        total_processing_seconds += record["total_seconds"]

        status = record.get("status", "unknown")

        if status not in status_counts:
            status = "unknown"

        status_counts[status] += 1


        if seconds is not None:
            generation_count += 1
            generation_total_seconds += seconds

    print("生成接口调用次数：", generation_count)
    print("未调用生成接口的次数：", len(records)-generation_count)
    print(f"生成接口累计耗时：{generation_total_seconds:.3f} 秒")

    if generation_count > 0:
        generation_average_seconds = generation_total_seconds / generation_count
        print(f"生成接口平均耗时：{generation_average_seconds:.3f} 秒")

    else:
        print("没有生成接口调用记录，无法计算平均耗时")

    if len(records) > 0:
        total_average_processing_seconds = total_processing_seconds / len(records)
        print(f"全部问题平均处理耗时：{total_average_processing_seconds:.3f} 秒")

    else:
        print("没有问题记录，无法计算平均处理耗时")

    print("各状态记录数：", status_counts)

if __name__ == "__main__":
    main()