from unittest.mock import MagicMock, patch
from openai import APIConnectionError, APIStatusError
from rag_assistant.agent import call_llm_with_retry
from httpx import Request

def test_first_retry_success():
    mock_client = MagicMock()
    mock_stream = MagicMock()
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=None)

    mock_request = Request("Post", "https://api.openaia.com/v1/chat/completions")

    mock_client.chat.completions.create.side_effect = [
        APIConnectionError(request=mock_request),
        mock_stream
    ]

    with patch("rag_assistant.agent.collect_stream_response") as mock_collect:
        mock_collect.return_value = {"content": "测试回答", "tool_calls": None}

        success, result, error_info = call_llm_with_retry(mock_client, messages=None, tool_definitions=None, tool_choice=None, on_text=None, max_retries=3)

        assert success == True
        assert result == {"content": "测试回答", "tool_calls": None}
        assert error_info == None
        assert mock_client.chat.completions.create.call_count == 2

def test_retry_exhausted():
    mock_client = MagicMock()

    mock_request = Request("Post", "https://api.openaia.com/v1/chat/completions")

    mock_client.chat.completions.create.side_effect = [
        APIConnectionError(request=mock_request),
        APIConnectionError(request=mock_request),
        APIConnectionError(request=mock_request),
    ]

    success, result, error_info = call_llm_with_retry(mock_client, messages=None, tool_definitions=None, tool_choice=None, on_text=None, max_retries=3)

    assert success == False
    assert result == None
    assert error_info["status"] == "api_error"
    assert "重试3次" in error_info["message"]
    assert mock_client.chat.completions.create.call_count == 3

def test_client_error_no_retry():
    mock_response = MagicMock()
    mock_response.status_code = 401

    error_401 = APIStatusError(
        message="Unauthorized",
        response=mock_response,
        body=None
    )

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = error_401

    success, result, error_info = call_llm_with_retry(mock_client, messages=None, tool_definitions=None, tool_choice=None, on_text=None)

    assert success == False
    assert result == None
    assert error_info["status"] == "api_error"
    assert "不可重试" in error_info["message"]
    assert "401" in error_info["message"]
    assert mock_client.chat.completions.create.call_count == 1

def test_server_error_retry():
    mock_response = MagicMock()
    mock_response.status_code = 500

    error_500 = APIStatusError(
        message="Server Error",
        response=mock_response,
        body=None
    )

    mock_stream = MagicMock()
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=None)
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [error_500, mock_stream]
    with patch("rag_assistant.agent.collect_stream_response") as mock_collect:
        mock_collect.return_value = {"content": "测试内容", "tool_calls": None}

        success, result, error_info = call_llm_with_retry(mock_client, messages=None, tool_definitions=None, tool_choice=None, on_text=None)

    assert success == True
    assert result != None
    assert error_info == None
    assert mock_client.chat.completions.create.call_count == 2




if __name__ == "__main__":
    print("运行测试1：第1次失败，第2次成功")
    test_first_retry_success()
    print("通过\n")

    print("运行测试2：重试3次都失败")
    test_retry_exhausted()
    print("通过\n")

    print("运行测试3：错误不重试")
    test_client_error_no_retry()
    print("通过\n")

    print("运行测试4：错误重试")
    test_server_error_retry()
    print("通过\n")

    print("所有测试都通过")