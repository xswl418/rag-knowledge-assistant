# 样例与参考来源

## 文档样例

`data/samples/` 内书屋、工作坊和设备借用资料均为虚构测试文档，不代表真实机构规则。

| 文件 | 用途 |
| --- | --- |
| `knowledge_base.txt` | 基础 RAG 默认资料；包含 RAG、工具调用、错误与评测相关简述 |
| `library_rules.txt` | 简短借阅规则 |
| `library_guide.txt` / `.docx` / `.pdf` | 标题、正文、借阅条件与来源检查；不同格式内容并非完全相同 |
| `library_pages_test.docx` / `.pdf` | 分页来源检查；PDF 第二页空白，后续页码仍应保持原值 |
| `workshop_notice.txt` | 报名、候补、条件限制与缺失字段 |
| `equipment_loan_notes.txt` | 设备借用及互相冲突的数量规则，不提供版本优先级 |
| `summary_long_test.txt` | 合成摘要长文，检查分批与最终汇总 |
| `extraction_long_test.txt` | 合成提取长文，跨批次放置相同及不同候选值 |

长文样例只用于控制流程验证，不代表真实长文质量。
提取长文可由 `python -m scripts.create_extraction_sample` 生成；已有文件与预期不同时脚本拒绝覆盖。
`data/eval_cases.json` 为开发用例，不包含运行结果或用户上传文档。

## 知识库来源

`knowledge_base.txt` 的部分概念材料依据以下官方资料整理，属于项目中文简述，不是官方译文，也不是当前平台规则的完整说明：

| 主题 | 原始资料 |
| --- | --- |
| 向量与语义检索 | OpenAI 文档：Vector embeddings、Retrieval |
| 工具调用流程与 schema | OpenAI 文档：Function calling |
| API 异常、限流与连接 | OpenAI 文档：Error codes |
| RAG 错误定位 | OpenAI 文档：Optimizing LLM Accuracy |
| 评测数据与持续回归 | OpenAI 文档：Evaluation best practices |
| 请求次数、并行与流式 | OpenAI 文档：Latency optimization |

检索阈值取舍不等于本地模型通用推荐值，第三方接口行为需按对应服务说明核对。
样例内的解释文字不应作为未经核验的生产配置或最新 API 规范。
