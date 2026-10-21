import os
from openai import OpenAI, APIStatusError, APIConnectionError


def create_api_client():
    api_key = os.environ["OPENAI_API_KEY"]
    base_url = os.environ["OPENAI_BASE_URL"]

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=120.0,
        max_retries=0,
    )


if __name__ == "__main__":
    from rag_assistant.rag import LLM_MODEL_NAME

    with create_api_client() as client:
        print("接口地址：", client.base_url)
        print("模型名称：", LLM_MODEL_NAME)

        try:
            response = client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": "请只回答：API连接成功",
                    }
                ],
            )
        except APIStatusError as error:
            print("HTTP 状态码：", error.status_code)
            print("服务返回：", error.body)
        except APIConnectionError as error:
            print("连接异常类型：", type(error).__name__)
            print("连接异常说明：", str(error))
        else:
            print("模型返回：", response.choices[0].message.content)
            print("结束原因：", response.choices[0].finish_reason)
