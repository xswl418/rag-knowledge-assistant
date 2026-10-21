import json

from rag_assistant.document_task import merge_field_results


def main():
    # Different values must retain their own evidence instead of being resolved.
    second_value = "10:00"
    target_field = "开放时间"

    # 手工构造三批成功提取后的结果，不调用模型。
    batch_results = [
        {
            "items": [
                {
                    "field": "开放时间",
                    "value": "09:00",
                    "evidence": "本馆每天09:00开放。",
                    "sources": [
                        {
                            "source": "sample.txt",
                            "chunk_id": 1,
                            "paragraph_id": 1,
                            "page_number": None,
                        }
                    ],
                }
            ]
        },
        {
            "items": [
                {
                    "field": "开放时间",
                    "value": second_value,
                    "evidence": "开放时间为10:00。",
                    "sources": [
                        {
                            "source": "sample.txt",
                            "chunk_id": 8,
                            "paragraph_id": 2,
                            "page_number": None,
                        }
                    ],
                }
            ]
        },
        {
            "items": [
                {
                    "field": "开放时间",
                    "value": None,
                    "evidence": None,
                    "sources": [],
                }
            ]
        },
    ]

    result = merge_field_results(batch_results, target_field)

    assert result["status"] == "multiple"
    assert len(result["options"]) == 2
    assert len(result["options"][0]["evidences"]) == 1

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
