# Streamlit Community Cloud 部署

本文提供本项目的部署配置和核验清单，不表示云端运行已经验收。
托管与生成模型 API 的费用分开计算，实际资源、下载和接口连通性需在部署环境确认。

## 仓库配置

| 配置 | 本项目值 |
| --- | --- |
| Repository | `xswl418/rag-knowledge-assistant`；使用副本时填写自己的仓库 |
| Branch | `main` |
| Main file path | `web_app/app.py` |
| Python version | 本地已验证 3.13 系列，优先使用相同系列 |

仓库应包含 `rag_assistant/`、`web_app/`、`requirements.txt` 和 `.streamlit/config.toml`。
不要上传密钥、环境目录、用户文档、运行日志或本地模型权重。

## Secrets

在应用设置中填写根级 TOML 配置，将占位符替换为应用实际使用的值：

```toml
OPENAI_API_KEY = "REPLACE_WITH_APP_API_KEY"
OPENAI_BASE_URL = "https://your-provider.example/v1"
EMBEDDING_MODEL_PATH = "BAAI/bge-small-zh-v1.5"
EMBEDDING_ALLOW_DOWNLOAD = "true"
```

应用通过环境变量读取这些值。`OPENAI_BASE_URL` 必须与密钥配套；
生成模型名称在 `rag_assistant/rag.py` 的 `LLM_MODEL_NAME` 中配置，需与服务商支持的名称一致。
本地 `.streamlit/secrets.toml` 已忽略，切勿把真实 Secrets 写进公开仓库。

云端使用模型名称而非 Windows 绝对路径。`EMBEDDING_ALLOW_DOWNLOAD` 默认为 `false`；
显式设为 `true` 后，网页允许按需下载并缓存向量模型。首次检索可能等待下载和加载；
平台重建或清理缓存后可能再次下载。更换模型或开关后应重启应用。
概括与字段提取不依赖检索资源，但仍需要调用生成接口。

## 部署核验

1. 确认依赖安装成功，页面无导入错误，主题和本地图标正常。
2. 上传公开样例并提出具体事实问题，核对模型加载、接口调用及来源。
3. 分别尝试全文概括和字段提取，确认完整结果或明确的失败提示。
4. 更换文档并清空对话，核对旧文档历史不会继续参与处理。
5. 用另一个浏览器会话检查文档和回答隔离；停止本地服务后仍能访问云端地址。
6. 分享地址前核对平台应用可见性、生成接口费用及日志中的敏感内容。

构建失败先看依赖日志，模型失败检查下载或加载错误，接口失败检查请求状态；
不要用反复重启代替定位首个错误。

本项目没有自行实现登录或调用配额。页面历史属于内存状态，实例本地 JSONL 文件不是持久化服务。
首次部署应使用公开测试资料；不要把云端试用当作敏感文档生产服务。

## 验证范围

本地离线回归覆盖网页任务流程、失败收尾、重复运行、批次进度及模型下载开关；
运行方法和边界见 [测试与评测](testing.md)。
离线测试使用模拟接口和模型，不能代替云端依赖、网络、内存及真实回答质量验收。
