# RAG 知识库问答与工具调用 Agent

一个逐步开发的 Python 命令行学习项目。使用本地向量模型检索文本知识库，并通过大模型的工具调用请求完成“选择检索工具、执行检索、读取资料、生成回答”的流程。

当前阶段已实现单工具 Agent、连续独立提问、异常处理、任务日志和小规模人工评测。知识库包含 14 个 AI 基础知识段落；下一阶段是接入用户指定的 TXT 文件。

## 当前功能

- 用 `BAAI/bge-small-zh-v1.5` 生成向量，以余弦相似度、Top-K 和最低分数筛选资料。
- 先按空行分段，长段落再按字符长度和重叠长度切块；每块保留文件名、段落编号和知识块编号。
- 模型通过 `search_knowledge` 请求检索，Python 校验工具名称和参数，执行函数并回传资料。
- 工具请求的 JSON 或参数不合法时，将错误交回模型，允许其在调用上限内修正。
- 每个问题最多尝试 3 次模型调用；每次响应只支持一条工具调用请求。
- 支持简单问候直接回答、空回答处理，以及超时、连接、鉴权、限流和 HTTP 状态异常提示。
- 输入 `exit` 退出；空输入得到提示；多个问题分别初始化消息、状态和计数。
- 将状态、模型调用次数、工具调用次数和总耗时保存为 JSONL，提供汇总脚本。

## 两个运行入口

| 入口 | 用途 | 每次提问的流程 |
| --- | --- | --- |
| `manual_test_tool_execution.py` | 当前 Agent 主入口，保留开发时的文件名 | 模型决定是否请求工具 → Python 执行并回传 → 模型继续处理或回答 |
| `rag.py` | 基础 RAG 入口，同时提供共享函数 | 程序固定检索 → 构建提示词 → 调用模型回答 |

两个入口都会在启动时读取知识库、加载本地向量模型并生成知识块向量，后续问题复用这些资源。

Agent 检索结果为空时，程序以 `no_results` 结束当前任务并等待新问题。此时通常已经调用模型来选择工具，但不会再请求模型生成答案。基础 RAG 入口则在检索无结果时完全跳过生成模型调用。

这里的连续提问不包含跨问题聊天记忆。每个问题独立处理，不能依赖上一题理解“它”“刚才那个”等追问。

## 安装与运行

已验证的开发环境：Windows、Python 3.13.14。以下命令在项目目录的 PowerShell 中执行。

### 1. 安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. 首次下载向量模型

这一步需要能够访问模型下载服务；模型已缓存时可以跳过。运行入口使用 `local_files_only=True`，不会自动下载缺失模型。

```powershell
.\.venv\Scripts\python.exe -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh-v1.5')"
```

### 3. 配置生成模型

在运行环境中设置 `OPENAI_API_KEY`。使用 OpenAI 兼容服务时，还需设置与密钥配套的 `OPENAI_BASE_URL`，并将 `rag.py` 中的 `LLM_MODEL_NAME` 改为服务商支持的模型名称。

Agent 要求接口支持 Chat Completions 的工具调用。配置名称不代表已独立核验服务商实际使用的后端模型。程序不会自动读取 `.env` 文件；不要把真实密钥写入代码或提交到仓库。

### 4. 启动 Agent

确保 `knowledge_base.txt` 位于项目目录，然后运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 -B manual_test_tool_execution.py
```

可尝试“RAG 是怎么工作的？”或“知识库助手能正常返回文字，但用户说回答不对。我应该怎样排查，并验证修改是否有效？”。输入小写 `exit` 退出，首尾空白会被去除。

基础 RAG 的运行命令为：

```powershell
.\.venv\Scripts\python.exe -X utf8 -B rag.py
```

## 知识库与参数

参数集中在 `rag.py`：

| 参数 | 当前值 | 含义 |
| --- | --- | --- |
| `MAX_CHUNK_SIZE` | `200` | 每块最多 200 个 Python 字符，不是 200 个 token |
| `CHUNK_OVERLAP` | `20` | 长段落切块时，相邻块的重叠字符数 |
| `TOP_K` | `3` | 最多返回 3 个知识块 |
| `MINIMUM_SCORE` | `0.55` | 返回结果的最低余弦相似度 |

当前 14 个自然段都短于 200 个字符，因此产生 14 个知识块，向量形状为 `(14, 512)`。这些配置是本项目的开发参数，不是通用最佳值。

知识库从 4 段扩充到 14 段。人工观察发现，100 字符切块时，同一段的相邻块可能占据多个检索位置；调整到 200 字符后，一条综合问题取回了评测、错误定位和修改验证三类资料。这是小规模开发观察，且两次模型生成的检索问题略有变化，不能将全部差异归因于块长，也不能据此推断总体质量提升。

新增正文的来源对照见 [知识库补充材料](docs/knowledge_expansion_v1.md)。运行时引用目前只有本地文件名和段落编号，不会自动返回官方网页链接。

## 测试与日志

以下脚本分别检查不同问题之间的状态隔离和工具参数校验，不请求真实生成接口，也不加载向量模型。状态隔离脚本带有断言，参数校验脚本需要人工对照输出：

```powershell
.\.venv\Scripts\python.exe -X utf8 -B manual_test_agent_state.py
.\.venv\Scripts\python.exe -X utf8 -B manual_test_tool_validation.py
```

基础 RAG 的空回答测试需要已缓存的向量模型，但生成回答使用模拟结果，只写入独立的测试日志：

```powershell
.\.venv\Scripts\python.exe -X utf8 -B manual_test_no_answer.py
```

`manual_test_tool_call.py` 和 `manual_eval_insufficient_context.py` 会调用真实模型，可能产生费用，不属于上述离线回归测试。后者保留了旧版严格知识库回答标准，用于历史对照，不代表未来的通用知识补充策略。

实际运行并生成日志后，可以查看汇总：

```powershell
.\.venv\Scripts\python.exe -X utf8 -B analyze_agent_logs.py
.\.venv\Scripts\python.exe -X utf8 -B analyze_logs.py
```

两个脚本分别读取 `agent_logs.jsonl` 和 `query_logs.jsonl`。日志尚不存在时，请先运行对应入口；汇总脚本目前不处理缺失或损坏的日志文件。日志已加入 `.gitignore`，不会随项目上传。

`answered` 只表示获得非空白回答，不代表答案正确。模型调用次数记录程序尝试发出的请求，包含失败尝试，不等于服务商计费次数。Agent 客户端设置 30 秒超时并关闭 SDK 自动重试。

## 当前边界与后续方向

- 当前读取固定的本地 TXT 文件，尚不支持用户选择文件、Word、PDF 或网页上传。
- 知识库与向量保存在内存中；修改资料后需重启，重启时重新生成向量。
- 切块尚不识别标题、句子边界、表格或页面；200 字符并不适合所有文档。
- 模型可以在一次任务内重复调用同一个检索工具，但尚未支持多工具协作或同轮多条工具请求。
- 引用和“依据资料回答”依赖模型表现，尚无自动事实核验；已有开发用例不能推断真实场景准确率。
- 当前运行策略在检索为空时停止。允许明确标注的通用知识补充已列入产品讨论，尚未完整实现。
- 暂无流式输出、跨问题记忆、Web 服务或多用户资料管理。

下一阶段按“用户导入 TXT → 真实文章的切块与来源定位 → Word → 文字型 PDF → 上传问答界面”推进，复杂版式和扫描件单独扩展。

## 开发记录

- [首批 RAG 评测记录（4 段知识库，历史版本）](docs/evaluation_v1.md)
- [Agent 资料不足评测及产品规则讨论](docs/agent_evaluation_v1.md)
- [知识库补充正文与来源](docs/knowledge_expansion_v1.md)

上述文档保留各自整理时的条件、结果和待办。知识库扩充状态及当前参数以本 README 和代码为准，历史记录不代表当前版本的完整评测。
