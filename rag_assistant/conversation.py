MAX_HISTORY_RECORDS = 5

def build_history_messages(records):
    history_messages = []

    if MAX_HISTORY_RECORDS <= 0 :
        return history_messages

    for record in records[-MAX_HISTORY_RECORDS:]:
        if record["status"] == "answered":
            history_messages.append({
                "role": "user",
                "content": record["question"],
            })
            history_messages.append({
                "role": "assistant",
                "content": record["answer"],
            })

    return history_messages