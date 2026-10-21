from scripts.analyze_eval_results import is_completed_run

assert is_completed_run({
    "status": "answered",
    "answer": "不能续借",
})

assert not is_completed_run({
    "status": "answered",
    "answer": " "
})

assert not is_completed_run({
    "status": "answered",
    "answer": None
})

assert not is_completed_run({
    "status": "api_error",
    "answer": "部分文字"
})

print("四项运行完成检查通过。")