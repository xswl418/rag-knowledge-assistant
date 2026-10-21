# 测试与评测

所有命令从仓库根目录执行。自动化回归、手动流程检查与真实模型内容评测分开记录。

## 离线自动化回归

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

- `test_web_streaming.py`：问答流式、检索、概括与提取进度、异常收尾、会话历史、重新运行不重复保存；共 26 项，包含 4 项实际网页加载函数的下载开关检查。
- `test_project_layout.py`：10 项检查，覆盖模块导入、根目录路径、静态资源、样例解析、批次计划、用例加载、源码语法及非项目工作目录下的网页启动。

2026-10-21 在独立 Python 3.13.14 环境中，整理后的 36 项自动化测试全部通过。

网页测试使用 Streamlit AppTest，模型客户端、向量模型及相关保存操作均采用替身。
测试不会发起真实生成请求或下载权重，也不修改日常任务日志。
没有真实接口参与的回归不能证明回答准确率、服务商稳定性或云端性能。

## 手动离线检查

```powershell
python -m tests.manual_test_extraction_merge
python -m tests.manual_test_stream_tool
python -m tests.manual_test_eval_run
python -m tests.manual_test_tool_validation
```

以上脚本分别检查提取候选合并、工具参数增量拼接、运行完成条件和工具参数校验。
`manual_test_document_loading`、`manual_test_docx_loading`、`manual_test_pdf_loading` 接收本地文件路径，供检查解析与来源。
手动脚本不计入自动化测试数量。

## 真实模型评测

以下操作会产生真实模型请求和费用，需要配套的 API 环境变量及已缓存的向量模型：

```powershell
python -m scripts.evaluate_documents
```

按提示输入一个或多个用例编号（如 `D01 D02`），或输入 `all`。
用例在 `data/eval_cases.json`，包含文档路径、问题、预期要点及可选前置历史；
问题覆盖三份虚构短文的事实、信息缺失、跨段条件、指代、冲突、概括和提取。

每批结果写入 `logs/eval_results.jsonl`，保存批次 ID、开始时间、检索与模型配置和完整任务记录。
运行完成只检查状态及非空答案，不自动判内容正确。

```powershell
python -m scripts.analyze_eval_results
```

按原文、预期要点、回答及来源进行人工 `pass/fail/pending` 复核，理由保存到 `logs/eval_reviews.jsonl`。
没有来源支撑、条件遗漏或把缺失字段补成事实均应单独检查，不能只用关键词命中评分。
这份数据属于开发集，不是独立留出集；不得将重复运行或人工开发检查换算成总体准确率。

## 接口与日志诊断

| 命令 | 行为 |
| --- | --- |
| `python -m rag_assistant.api_client` | 一次普通生成请求，检查基础连通性；不验证工具调用 |
| `python -m scripts.manual_test_stream` | 一次真实流式工具选择请求；只验证并打印参数，不执行工具 |
| `python -m rag_assistant.tools` | 一次非流式工具选择请求；不执行检索 |
| `python -m scripts.manual_eval_insufficient_context` | 固定返回相关却不足的资料，人工核对真实回答的边界 |
| `python -m scripts.analyze_agent_logs` | 汇总本地 Agent 日志，不调用模型 |
| `python -m scripts.analyze_logs` | 汇总基础 RAG 日志，不调用模型 |

诊断输出和日志可能包含服务地址、原文或回答，不应原样上传公开仓库。
开发演进及已知失败见 [变更记录](../CHANGELOG.md)，云端验收清单见 [部署说明](community_cloud.md)。
