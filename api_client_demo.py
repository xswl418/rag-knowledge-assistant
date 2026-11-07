def display_result(data):
    if data['status'] == "answered":
        print(data["answer"])
    else:
        print(data["status"])
        if data["error_message"]:
            print(data["error_message"])


success_data = {
    "question": "每位读者最多能同时借几本书，借阅期限是多少？",
    "status": "answered",
    "answer": "最多三本，每本十二天。来源：library_guide.txt，第2组。",
    "error_message": None,
}

failure_data = {
    "question": "每位读者最多能同时借几本书，借阅期限是多少？",
    "status": "api_error",
    "answer": None,
    "error_message": "模型接口请求超时。",
}

display_result(success_data)
display_result(failure_data)