# 文档问答与分析 Agent

上传一份文档，用自然语言查询条款、概括全文或提取指定字段，并回到来源核对结果。
项目基于 Python、Streamlit 和支持工具调用的 Chat Completions 兼容接口，提供网页、命令行和固定样例的本地HTTP API入口。

在线体验：<https://rag-knowledge-assistant-kufyplzg5amhfd3sxyc5rt.streamlit.app/>

## 功能

- **文档问答**：支持 UTF-8 TXT、DOCX 和文字型 PDF；本地向量检索返回相关片段，回答可附文件名、分组或 PDF 页码。
- **全文概括**：按全文可提取文字处理，长文分批生成后汇总，展示批次进度；超限或中途失败时明确提示，不把部分结果作为完整摘要。
- **字段提取**：按指定字段逐批提取，校验原文引文并合并来源；未找到的字段标记缺失，不同候选值并列展示，不自动裁决。
- **连续追问**：网页使用当前文档最近有限条成功问答理解指代；清空对话或切换文档会重置历史。
- **流式与可追溯结果**：问答和概括支持流式展示，可展开检索资料；任务状态、调用次数、耗时等写入本地 JSONL 日志。

网页仅在实际检索时准备向量资源。单独概括或提取不需要先加载向量模型，但仍需要生成模型接口。

## 快速开始

以下命令从项目根目录执行，示例使用 Windows PowerShell。本地验证环境为 Windows、Python 3.13.14；其他操作系统仍需验证依赖与模型加载。

### 1. 安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. 配置模型接口

在启动应用的同一终端设置环境变量：

```powershell
$env:OPENAI_API_KEY = "替换为应用使用的密钥"
$env:OPENAI_BASE_URL = "https://your-provider.example/v1"
```

生成模型默认使用 `gpt-6.1-sol`。也可以通过 `LLM_MODEL_NAME` 环境变量覆盖为服务商实际支持的模型名称，
无需修改源码；例如在云端 Secrets 中增加：

```toml
LLM_MODEL_NAME = "服务商提供的快速模型名称"
```

当前默认值不代表对第三方服务实际后端模型的独立验证。
接口需支持工具调用和流式响应；普通文本请求成功不代表这两项能力均可用。

应用同时读取上述两个变量，**不会自动加载 `.env` 文件**。不要把真实密钥写入源码或提交到 Git。

### 3. 准备向量模型

默认使用 `BAAI/bge-small-zh-v1.5`。首次使用时，可先下载到本地缓存：

```powershell
.\.venv\Scripts\python.exe -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh-v1.5')"
```

这一步需要连接模型下载服务。已有缓存则不必重复执行。
也可通过 `EMBEDDING_MODEL_PATH` 指定完整模型目录，或在网页启动前设置 `EMBEDDING_ALLOW_DOWNLOAD="true"` 允许网页按需下载；未设置时仅使用本地模型。
命令行入口和评测脚本始终只读取本地模型。

### 4. 启动网页

```powershell
.\.venv\Scripts\python.exe -m streamlit run web_app/app.py
```

打开终端显示的 Local URL。上传文档，待“文档已就绪”后提交问题，例如：

- 问答：“每位读者最多能借几本书，借阅期限是多少？”
- 概括：“请概括这份文档的主要内容。”
- 提取：“请提取借阅上限、借阅期限、续借次数和联系电话。”

可直接使用 `data/samples/library_guide.txt` 体验。样例为虚构资料，不代表真实机构规定。
回答下方可展开检索资料并核对来源；检索返回的内容不一定全部被模型实际引用。

Windows 也可双击根目录的 `start_web.bat`，它使用项目 `.venv` 启动同一网页。
通过环境变量配置时，请在已经设置变量的终端运行启动器；单独双击不会继承另一个终端临时设置的变量。
本地服务运行期间需保持启动窗口开启，关闭服务后仅打开网址不会自动启动应用。

## 配置与限制

| 配置位置 | 默认值或行为 |
| --- | --- |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | 必填；需为同一服务配套的密钥与接口地址 |
| `EMBEDDING_MODEL_PATH` | `BAAI/bge-small-zh-v1.5`，也可设置完整本地模型目录 |
| `EMBEDDING_ALLOW_DOWNLOAD` | 默认 `false`；设为 `true` 允许网页下载模型，修改后需重启应用 |
| `rag_assistant/rag.py` | 切块 200 字符、重叠上限 20、Top-K 3、最低相似度 0.53 |
| `rag_assistant/conversation.py` | 从最近 5 条任务记录中选择成功问答作为网页历史；不按 token 截断 |
| `rag_assistant/document_task.py` | 每批最多 12000 字符（含来源标记），最多 10 批 |
| `rag_assistant/api_client.py` | 请求超时 120 秒、SDK 自动重试 0 次；不是整项任务的总时限 |

普通句子换块时，在容纳新句的剩余空间内保留上一块尾部文字，最多保留配置的重叠字符数；
空间不足时减少重叠，没有空间时不重叠。尾部文字可能不是完整句子，章节或页面之间不保证重叠。

一次任务最多执行 3 次检索、4 次外层模型请求，含工具内部请求的总上限为 15 次。
首次检索为空时允许模型改写查询；第三次仍为空则停止。预算或接口异常也可能提前终止任务。
概括与提取会先检查完整计划所需预算，不会为了完成而静默截掉后续批次。

## 检索评测

使用 40 道人工标注问题（基于 `data/samples/library_guide.txt`），评测指标为 Recall@3 和 MRR：

| 方法 | Recall@3 | MRR | 命中数 |
| --- | --- | --- | --- |
| 向量检索（bge-small-zh-v1.5） | 0.950 | 0.950 | 38/40 |
| BM25（rank-bm25 + jieba） | 0.925 | 0.850 | 37/40 |
| RRF 融合（向量 + BM25） | **1.000** | **0.950** | **40/40** |

RRF 融合补上了向量检索和 BM25 各自漏掉的题目。评测脚本见 `scripts/evaluate_retrieval.py`，标注数据见 `data/eval_questions.jsonl`。

- 一次处理一份文档；尚不支持 OCR、跨文档联合检索或持久化向量库。
- DOCX 主要读取正文段落与标题样式；PDF 读取已有文字层，扫描件、复杂排版、表格及图像内容不在可靠解析范围内。
- 向量阈值和切块参数不是通用最优值。检索可能漏掉依据或带入无关片段，生成内容仍需核对。
- 会话保存在内存中，刷新或重启可能丢失；本地日志不会自动恢复为页面历史。
- 当前没有项目级登录、访问控制或调用配额管理，部署后应核对平台可见性和费用风险。

## 数据与隐私

向量计算在应用运行机器上完成。处理任务时，问题、有限历史及选中的文档内容会发往配置的生成模型服务；概括和提取可能逐批发送全文。
请只上传有权处理的资料，并先确认服务商的数据处理政策。

`logs/` 可能包含问题、完整回答和原文片段，不是匿名统计。密钥、日志、虚拟环境和本地模型目录均被 Git 忽略；云端实例本地日志不作为长期存储。

## 其他入口

本地HTTP API支持固定样例问答、SQLite结果保存和按编号/状态查询，使用独立的`requirements-api.txt`安装清单；配置、启动和协议见[HTTP API说明](docs/api.md)。

```powershell
# 命令行 Agent：按提示选择一个 TXT、DOCX 或 PDF，输入 exit 退出
.\.venv\Scripts\python.exe -m rag_assistant.agent

# 基础 RAG：固定读取 data/samples/knowledge_base.txt
.\.venv\Scripts\python.exe -m rag_assistant.rag

# 接口连通性检查：会产生一次真实模型请求
.\.venv\Scripts\python.exe -m rag_assistant.api_client

# 离线自动化测试：不请求模型、不下载权重
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

命令行 Agent 每个问题独立处理，不传网页会话历史。
批量评测会产生真实模型请求，运行与人工复核方法见 [测试与评测](docs/testing.md)。

## 项目结构

```text
rag-knowledge-assistant/
├── rag_assistant/        # Agent、文档处理、检索、工具及共享客户端
├── web_app/              # Streamlit 入口及本地静态资源
├── scripts/              # 批量评测、日志分析和接口诊断
├── tests/                # 离线自动化测试及手动检查
├── data/                 # 评测用例和虚构样例文档
├── docs/                 # 架构、测试、数据来源和部署说明
├── .streamlit/           # 网页主题配置
├── api_app.py            # 固定样例HTTP API与SQLite任务保存查询
├── requirements.txt
├── requirements-api.txt
├── start_web.bat
├── CHANGELOG.md
└── README.md
```

旧版根目录模块已整理到 `rag_assistant/`，命令行改用 `python -m rag_assistant.agent`；
网页入口和样例位置保持不变。

- [开发演进](CHANGELOG.md)：历史提交、功能变化与验证边界。
- [设计与验证记录](docs/development.md)：近期迭代的问题、方案、修复和验证依据。
- [架构说明](docs/architecture.md)：请求链路、模块职责与数据路径。
- [HTTP API说明](docs/api.md)：本地API依赖、资源准备、启动、任务保存与查询。
- [测试与评测](docs/testing.md)：离线回归与真实模型评测的区别。
- [样例与来源](docs/sample_data.md)：测试文档用途和知识库参考来源。
- [Community Cloud 部署](docs/community_cloud.md)：仓库入口、Secrets 配置、冷启动说明和上线核验清单。
