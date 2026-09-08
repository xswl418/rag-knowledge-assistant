# RAG 知识库问答助手

一个 Python 命令行学习项目：读取本地文本知识库，通过向量检索找到相关资料，再调用大模型生成回答。

## 功能

- 按段落切分文档，对长段落进行带重叠的字符切分。
- 使用向量相似度、Top-K 和最低分数筛选资料。
- 将参考资料与用户问题组合成提示词，要求模型注明来源。
- 没有知识块达到阈值时，直接提示未找到相关内容。
- 对部分常见 API 错误提供中文提示。

## 运行方法

开发环境：Windows，Python 3.13.14。
以下命令在项目目录的 PowerShell 中执行。

### 1. 创建环境并安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. 首次下载向量模型

这一步需要能够访问模型下载服务。模型已缓存时可以跳过。

```powershell
.\.venv\Scripts\python.exe -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh-v1.5')"
```

### 3. 配置模型接口

在运行环境中设置：

- OPENAI_API_KEY：接口服务商提供的 API Key。
- OPENAI_BASE_URL：与该 Key 配套的 OpenAI 兼容 API 基础地址。

将 app.py 中的 LLM_MODEL_NAME 改为服务商支持的模型名称。
程序使用 Chat Completions 接口；不会自动读取 .env 文件。
不要在代码或说明文档中填写真实密钥。

### 4. 启动程序

确保 knowledge_base.txt 与 app.py 位于同一目录。

```powershell
.\.venv\Scripts\python.exe app.py
```

## 已验证示例

- “API请求失败时应该检查什么？”：根据知识库第4段回答并标注来源。
- “如何计算定积分？”：提示没有找到足够相关的内容，不调用生成接口。

## 当前限制

- 演示知识库只有4个段落，测试规模较小。
- 向量保存在内存中，每次启动都会重新生成。
- 阈值不能保证检索一定正确，需要结合实际资料评估。
- 提示词要求模型依据资料回答，但不能保证完全没有编造。
- 当前每次运行只处理一个问题。