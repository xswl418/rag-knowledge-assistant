from manual_test_tool_call import validate_tool_request

print("正常请求：")
result = validate_tool_request(
    "search_knowledge", {"question": "RAG是什么？"}
)
print("检查结果：", result)

print("\n错误工具名称：")
result = validate_tool_request(
    "search_documents", {"question": "RAG是什么？"}
)
print("检查结果：", result)

print("测试请求1：")
result = validate_tool_request(
    "search_knowledge", {"question": "   "}
)
print("检查结果：", result)

print("测试请求2：")
result = validate_tool_request(
    "search_knowledge", []
)
print("检查结果：", result)

print("测试请求3：")
result = validate_tool_request(
    "search_knowledge", {}
)
print("检查结果：", result)