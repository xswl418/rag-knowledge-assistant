# RAG 文档问答与工具调用 Agent

一个基于 Python 的命令行文档问答项目。支持导入本地 TXT、DOCX 和文字型 PDF 文件，使用本地向量模型检索文档，并通过大模型的工具调用请求完成“选择检索工具、执行检索、读取资料、生成回答”的流程。

当前版本实现了文档读取、按标题分组、优先保留句子边界的切块、单工具 Agent 调用循环、连续独立提问、异常处理与任务日志。项目附带示例文档、手工测试脚本和小规模评测记录，尚未覆盖复杂文档解析及生产部署。

## 当前功能

- 用 `BAAI/bge-small-zh-v1.5` 生成向量，以余弦相似度、Top-K 和最低分数筛选资料。
- 启动时指定一个本地文件：TXT 使用 UTF-8 解码，DOCX 使用 `python-docx` 提取正文段落和样式，PDF 使用 `pypdf` 逐页提取已有文字。
- TXT 按“一、”“二、”等中文编号标题分组，无匹配标题时按空行分段；DOCX 按 `Heading 1` 样式分组，无该样式时保留非空正文段落。
- 长段落按中文句末标点拆句并组合成块，单句超长时按字符切分；每块保留文件名、分组编号和知识块编号。
- PDF 按页分别分组和切块，保留原始页序号供回答引用；未提取到文字的页面会提示并跳过，后续页码不会重新编号。
- 模型通过 `search_knowledge` 请求检索，Python 校验工具名称和参数，执行函数并回传资料。
- 工具请求的 JSON 或参数不合法时，将错误交回模型，允许其在调用上限内修正。
- 每个问题最多尝试 3 次模型调用；每次响应只支持一条工具调用请求。
- 支持简单问候直接回答、空回答处理，以及超时、连接、鉴权、限流和 HTTP 状态异常提示。
- 输入 `exit` 退出；空输入得到提示；多个问题分别初始化消息、状态和计数。
- 对空路径、文件不存在、不支持的扩展名、TXT 编码错误、没有可用文字及部分 DOCX、PDF 读取错误给出提示，并允许重新选择。
- 将当前文件名、任务状态、模型调用次数、工具调用次数和总耗时保存为 JSONL，提供汇总脚本。

## 两个运行入口

| 入口 | 用途 | 每次提问的流程 |
| --- | --- | --- |
| `agent.py` | 当前 Agent 命令行主入口 | 模型决定是否请求工具 → Python 执行并回传 → 模型继续处理或回答 |
| `rag.py` | 基础 RAG 入口，同时提供共享函数 | 程序固定检索 → 构建提示词 → 调用模型回答 |

Agent 入口读取启动时指定的 TXT、DOCX 或 PDF 文件；`rag.py` 入口仍读取项目内的 `data/samples/knowledge_base.txt`。两个入口都会加载本地向量模型并生成知识块向量，后续问题复用这些资源。

文档处理与问答流程如下：

```text
选择本地 TXT / DOCX / PDF
  → 提取正文与分组 → 切块 → 生成向量
  → 输入问题 → 模型返回回答或工具请求
  → Python 校验并执行检索 → 回传检索结果 → 模型继续处理
  → 输出结果并记录任务日志
```

Agent 检索结果为空时，程序以 `no_results` 结束当前任务并等待新问题。此时通常已经调用模型来选择工具，但不会再请求模型生成答案。基础 RAG 入口则在检索无结果时完全跳过生成模型调用。

这里的连续提问不包含跨问题聊天记忆。每个问题独立处理，不能依赖上一题理解“它”“刚才那个”等追问。

## 项目结构

```text
rag-knowledge-assistant/
  agent.py                    # Agent 主入口、文件选择与任务循环
  rag.py                      # 文档解析、切块、检索及基础 RAG 入口
  tools/
    __init__.py
    manual_test_tool_call.py   # 工具定义、函数映射、参数校验及调用示例
  tests/                      # 文档加载、状态隔离及参数检查脚本
    __init__.py
    manual_test_*.py
  scripts/                    # 日志汇总与受控评测脚本
    __init__.py
    analyze_logs.py
    analyze_agent_logs.py
    manual_eval_insufficient_context.py
  data/samples/               # TXT、DOCX、PDF 示例资料
  docs/                       # 历史评测与资料来源
  logs/                       # 运行时创建，已忽略，不提交
  requirements.txt
  README.md
```

所有命令均从项目根目录执行。子目录中的脚本使用 `python -m 包名.模块名` 启动，使其能够导入根目录中的共享模块。使用 PyCharm 打开上层目录时，应将 `rag-knowledge-assistant` 标记为 Sources Root，并选择本项目的 `.venv` 解释器。

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

在项目目录运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 -B agent.py
```

按提示输入 TXT、DOCX 或 PDF 文件路径。支持完整路径，以及相对于当前工作目录的路径；路径两侧的双引号会被去除。例如，在项目目录启动后可输入 `data/samples/library_guide.docx`，加载完成后提问：“每位读者最多能借几本书，借阅期限是多少？”

输入 `data/samples/library_pages_test.pdf` 可以检查 PDF 页码：该示例第 2 页为空白页，借阅规则位于第 3 页，跳过空白页后仍以第 3 页作为来源。

文件选择和提问阶段均可输入小写 `exit` 退出。每次运行加载一个文件，更换文件需重新启动程序。

基础 RAG 的运行命令为：

```powershell
.\.venv\Scripts\python.exe -X utf8 -B rag.py
```

## 示例文档与切块参数

以下文件均位于 `data/samples/`：

| 文件 | 内容 | 默认配置下的知识块数量 |
| --- | --- | --- |
| `knowledge_base.txt` | 14 段 AI、RAG、工具调用及评测基础材料 | 14 |
| `library_rules.txt` | 书屋开放、借阅与续借规则 | 3 |
| `library_guide.txt` | 包含中文编号标题的书屋使用指南 | 4 |
| `library_guide.docx` | 包含标题样式的简短 Word 示例 | 2 |
| `library_guide.pdf` | 简短书屋指南的单页文字型 PDF | 1 |
| `library_pages_test.docx` | 分页测试使用的 Word 原文 | 2 |
| `library_pages_test.pdf` | 共 3 页，第 2 页空白，用于检查跳页后的来源页码 | 2 |

书屋资料用于功能验证，不代表真实机构规则。TXT 与 DOCX 指南的内容和篇幅不同，并非同一文档的格式转换版本。

参数集中在 `rag.py`：

| 参数 | 当前值 | 含义 |
| --- | --- | --- |
| `MAX_CHUNK_SIZE` | `200` | 每块最多 200 个 Python 字符，不是 200 个 token |
| `CHUNK_OVERLAP` | `20` | 整句重叠的长度上限；单句超长时使用的重叠字符数 |
| `TOP_K` | `3` | 最多返回 3 个知识块 |
| `MINIMUM_SCORE` | `0.55` | 返回结果的最低余弦相似度 |

分组超过块长时，程序优先在 `。！？` 处拆句，再将完整句子组合成块。上一块的末句不超过重叠上限、且加入下一块后不超长时，才保留该整句作为重叠内容；因此，相邻块不保证都有 20 个字符的重叠。单句本身超过块长时，退回字符切分。

`paragraph_id` 表示程序处理后的分组序号，不是 Word 原始段落号或页码。PDF 中该编号每页重新计数，`chunk_id` 则在整份文件中连续编号；`page_number` 是从 1 开始的 PDF 页面顺序，不一定等于正文印刷页码。Agent 提示模型对 PDF 使用页码引用，对 TXT、DOCX 使用分组编号引用。标题不会自动复制到同一组的后续知识块。

这些配置是当前开发基线，不是通用最佳值。`TOP_K=3` 表示最多返回 3 块，低于阈值的候选仍会被过滤。

已知检索局限：在 `library_guide.txt` 上询问“小组讨论区需要提前多久预约、最多使用多长时间、迟到二十分钟会怎样”时，曾仅检索到迟到处理条款，预约时间与使用时长所在块被 `0.55` 阈值过滤。当前保留这一配置和失败案例，后续需通过多类问题比较检索覆盖情况，而非根据单题结果确定参数。

`knowledge_base.txt` 的补充材料来源见 [知识库补充材料](docs/knowledge_expansion_v1.md)。运行时不会自动返回原始网页链接或 Word 页码。

## 测试与日志

文档加载脚本用于人工检查分组、块长和内容，不加载向量模型，也不请求生成接口：

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m tests.manual_test_document_loading
.\.venv\Scripts\python.exe -X utf8 -B -m tests.manual_test_docx_loading
.\.venv\Scripts\python.exe -X utf8 -B -m tests.manual_test_pdf_loading
```

通用加载脚本支持三种格式，复用 Agent 的输入校验和重新选择流程；DOCX 脚本保留 TXT、DOCX 的直接解析示例；PDF 脚本打印知识块编号、来源页码和内容。文件路径示例：`data/samples/library_pages_test.pdf`。

以下脚本分别检查不同问题之间的状态隔离和工具参数校验，不请求真实生成接口，也不加载向量模型。状态隔离脚本带有断言，参数校验脚本需要人工对照输出：

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m tests.manual_test_agent_state
.\.venv\Scripts\python.exe -X utf8 -B -m tests.manual_test_tool_validation
```

基础 RAG 的空回答测试需要已缓存的向量模型，但生成回答使用模拟结果，只写入独立的测试日志：

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m tests.manual_test_no_answer
```

`tools/manual_test_tool_call.py` 和 `scripts/manual_eval_insufficient_context.py` 会调用真实模型，可能产生费用，不属于上述离线回归测试。运行时分别使用 `-m tools.manual_test_tool_call` 和 `-m scripts.manual_eval_insufficient_context`。后者保留了旧版严格知识库回答标准，用于历史对照，不代表未来的通用知识补充策略。

实际运行并生成日志后，可以查看汇总：

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m scripts.analyze_agent_logs
.\.venv\Scripts\python.exe -X utf8 -B -m scripts.analyze_logs
```

两个脚本分别读取 `logs/agent_logs.jsonl` 和 `logs/query_logs.jsonl`。保存首条记录时会自动创建日志目录；空回答测试使用 `logs/query_logs_mock.jsonl`。日志尚不存在时，请先运行对应入口；汇总脚本目前不处理缺失或损坏的日志文件。`logs/` 已加入 `.gitignore`，不会随项目上传。

`answered` 只表示获得非空白回答，不代表答案正确。模型调用次数记录程序尝试发出的请求，包含失败尝试，不等于服务商计费次数。Agent 客户端设置 30 秒超时并关闭 SDK 自动重试。

Agent 日志的 `source` 表示当前加载的文件，不代表该问题一定执行过检索。已有验证包含 7 个示例文件的读取与切块、PDF 空白页跳过和原始页序号保留、状态隔离、工具参数校验，以及人工端到端问答检查；尚无覆盖多类真实文档的系统评测结果。

## 当前边界与后续方向

- 当前支持单个本地 UTF-8 TXT、DOCX 或文字型 PDF 文件，尚不支持旧版 `.doc`、扫描件 OCR、网页上传或多文件联合检索。
- 知识库与向量保存在内存中；修改资料后需重启，重启时重新生成向量。
- TXT 标题识别依赖有限的中文编号规则，编号列表可能被当作标题；DOCX 仅识别 `Heading 1` 分组，不根据字号、加粗或视觉排版推断标题。
- DOCX 当前只提取正文段落，不提取表格、文本框、页眉页脚或图片文字。部分无效 DOCX 有错误提示，但未覆盖所有损坏文件类型。
- PDF 依赖文件中已有的文字层，不识别图片文字，不还原复杂表格和多栏版式；按页分别处理，不合并跨页句子。没有提取到文字不一定代表该页是扫描件，需核对原文件。当前不提供密码输入或解密流程，也未覆盖所有 PDF 读取异常。
- 分句仅处理指定的中文句末标点，单句过长时仍可能被截断；当前块长与阈值不适合所有文档。
- 模型可以在一次任务内重复调用同一个检索工具，但尚未支持多工具协作或同轮多条工具请求。
- 引用和“依据资料回答”依赖模型表现，尚无自动事实核验；已有开发用例不能推断真实场景准确率。
- 当前运行策略在检索为空时停止。允许明确标注的通用知识补充已列入产品讨论，尚未完整实现。
- 暂无流式输出、跨问题记忆、Web 服务或多用户资料管理。

后续方向包括可拖拽文件的上传问答界面、对外部署、全文概括与关键信息提取，以及基于固定问题集的检索与回答质量评测。复杂版式、表格和扫描件解析将单独扩展。

## 开发记录

- [首批 RAG 评测记录（4 段知识库，历史版本）](docs/evaluation_v1.md)
- [Agent 资料不足评测及产品规则讨论](docs/agent_evaluation_v1.md)
- [知识库补充正文与来源](docs/knowledge_expansion_v1.md)

上述文档保留各自整理时的条件、结果和待办。知识库扩充状态及当前参数以本 README 和代码为准，历史记录不代表当前版本的完整评测。
