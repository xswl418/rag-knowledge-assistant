from pathlib import Path
from scripts.analyze_logs import load_records

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = BASE_DIR / "logs" / "agent_logs.jsonl"

def calculate_model_seconds(record):
    timings = record.get("model_timings")

    if timings is None:
        return None

    model_seconds = 0.0

    for timing in timings:
        model_seconds += timing["seconds"]

    return model_seconds


def main():
    records = load_records(LOG_PATH)
    print("Agent任务记录数：", len(records))

    if records :
        latest_record = records[-1]
        model_seconds = calculate_model_seconds(latest_record)
        total_seconds = latest_record.get("total_seconds")

        if model_seconds is None:
            print("最近一次任务没有模型记时记录")
        elif total_seconds is None:
            print("最近一次任务没有总耗时记录")
        else:
            other_seconds = total_seconds - model_seconds

            print(f"最近一次任务总耗时：{total_seconds:.3f}秒")
            print(f"模型请求总耗时：{model_seconds:.3f}秒")
            print(f"任务内其余耗时：{other_seconds:.3f}秒")

    tool_record_count = 0
    tool_call_total = 0

    status_counts ={
        "answered": 0,
        "no_answer": 0,
        "no_results": 0,
        "api_error": 0,
        "max_calls": 0,
        "unsupported_tool_calls": 0,
        "unknown": 0
    }

    time_record_count = 0
    total_task_seconds = 0.0

    model_record_count = 0
    model_call_total = 0

    for record in records:
        status = record.get("status", "unknown")

        if status not in status_counts:
            status = "unknown"

        status_counts[status] += 1

        tool_calls = record.get("tool_call_count")

        if tool_calls is not None:
            tool_record_count += 1
            tool_call_total += tool_calls

        seconds = record.get("total_seconds")

        if seconds is not None:
            time_record_count += 1
            total_task_seconds += seconds

        model_calls = record.get("model_call_count")

        if model_calls is not None:
            model_record_count += 1
            model_call_total += model_calls


    print("已记录工具次数的任务数：", tool_record_count)
    print("缺少工具次数的任务数：", len(records) - tool_record_count)
    print("各状态任务数：", status_counts)

    print("已记录耗时的任务数：", time_record_count)
    print("缺少耗时的任务数：", len(records) - time_record_count)

    print("已记录模型的任务数：", model_record_count)
    print("缺少模型次数的任务数", len(records) - model_record_count)
    print("模型调用累计次数：", model_call_total)

    if time_record_count > 0:
        average_seconds = total_task_seconds / time_record_count
        print(f"已记录任务的平均耗时：{average_seconds:.3f}秒")

    else:
        print("没有耗时记录，无法计算平均值")

    if tool_record_count > 0:
        average_tool_calls = tool_call_total / tool_record_count
        print(f"已记录任务的平均工具调用次数：{average_tool_calls:.2f}")
    else:
        print("没有工具调用次数记录，无法计算平均值")

    if model_record_count > 0:
        average_model_calls = model_call_total / model_record_count
        print(f"已记录任务的平均模型调用次数：{average_model_calls:.2f}")
    else:
        print("没有模型调用次数记录，无法计算平均值")


if __name__ == "__main__":
    main()