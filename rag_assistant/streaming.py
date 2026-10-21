def merge_tool_call_delta(tool_calls, delta):
    index = delta["index"]

    if index not in tool_calls:
        tool_calls[index] = {
            "id": "",
            "type": "function",
            "function": {
                "name": "",
                "arguments": "",
            },
        }

    tool_call = tool_calls[index]
    function_delta = delta.get("function") or {}

    tool_call["id"] += delta.get("id") or ""
    tool_call["function"]["name"] += function_delta.get("name") or ""
    tool_call["function"]["arguments"] += function_delta.get("arguments") or ""

def collect_stream_response(stream, on_text=None):
    answer_parts = []
    tool_calls = {}
    finish_reason = None

    for chunk in stream:
        if not chunk.choices:
            continue

        choice = chunk.choices[0]
        content = choice.delta.content

        if content:
            answer_parts.append(content)

            if on_text is not None:
                on_text(content)

        for tool_delta in choice.delta.tool_calls or []:
            merge_tool_call_delta(
                tool_calls,
                tool_delta.model_dump(exclude_none=True),
            )

        if choice.finish_reason is not None:
            finish_reason = choice.finish_reason

    collected_tools = []

    for index in sorted(tool_calls):
        collected_tools.append(tool_calls[index])

    return {
        "content": "".join(answer_parts),
        "tool_calls": collected_tools,
        "finish_reason": finish_reason,
    }