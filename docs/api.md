# HTTP API

API入口为根目录的`api_app.py`，复用`rag_assistant.agent.run_agent`。当前读取固定公开合成样例`data/samples/library_guide.txt`，同步等待任务结束后保存最终结果，并提供任务查询。

API与原网页使用不同入口；网页会话不会自动从此任务数据库恢复。当前API未提供文档上传、会话历史、后台任务或HTTP流式响应。

## 安装依赖

以下命令从项目根目录运行，使用Windows PowerShell和已安装的Python：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-api.txt
```

已有项目环境时可直接使用该环境。`requirements-api.txt`引用原`requirements.txt`并增加FastAPI和Uvicorn，包含原网页依赖。此清单尚未完成独立干净环境重建验证。

## 配置与本地文件

在用于启动API的同一终端中设置：

```powershell
$env:OPENAI_API_KEY = "替换为应用使用的密钥"
$env:OPENAI_BASE_URL = "https://your-provider.example/v1"
```

接口客户端从进程环境读取这些配置，不会自动加载`.env`。配置存在或客户端对象创建成功，并不等于模型接口已通过鉴权或问答验证。

当前API在应用启动时读取项目内`models/bge-small-zh-v1.5`目录，使用`local_files_only=True`加载向量模型，并计算固定样例的块向量。请先准备该完整模型目录；安装Python依赖不会自动生成这里的权重文件。API此处使用固定项目路径，不通过网页的`EMBEDDING_MODEL_PATH`配置切换目录。

## 启动

```powershell
.\.venv\Scripts\python.exe -m uvicorn api_app:app --host 127.0.0.1 --port 8001 --reload
```

`api_app:app`表示导入`api_app.py`中的`app`对象；从其他目录执行可能导致模块导入失败。等待`Application startup complete`后再请求。`--reload`用于本地开发，保存代码会重载资源；终端保持开启，按Ctrl+C停止。

`127.0.0.1`仅监听本机。若端口已占用，请先确认原服务，或使用其他空闲端口并相应调整请求地址。

交互文档与只读健康检查地址：

```text
http://127.0.0.1:8001/docs
http://127.0.0.1:8001/health
```

健康检查返回文档块与向量数量，不验证生成模型鉴权或回答质量。

## 请求与响应

| 方法与路径 | 用途 |
| --- | --- |
| GET /health | 查看启动后资源数量 |
| POST /ask | 对固定样例提问，保存最终任务结果并返回编号 |
| GET /tasks/{task_id} | 根据编号读取一条已保存结果 |
| GET /tasks | 按编号降序查询已有结果 |
| GET /task-stats | 统计全部已保存任务的总数与各状态数量 |

POST请求正文：

```json
{"question": "每位读者最多能同时借几本书，借阅期限是多少？"}
```

`question`必填，长度为1至1000。POST会调用配置的模型服务。正常返回的任务响应包含`id`、`question`、`status`、`answer`、`error_message`；成功可没有错误说明，失败可没有答案，对应JSON空值为`null`。

当前协议下，Agent正常返回的业务失败也以HTTP200返回并保存，需要检查正文`status`；HTTP200不等于已生成正确答案。输入校验失败按框架返回422。POST保存环节抛出`sqlite3.Error`时返回HTTP500及`{"detail": "任务结果保存失败"}`，不返回任务编号；不能仅凭此响应判断具体数据库故障原因或推断一定存在可查询记录。其他未捕获的程序错误不保证产生可查询记录。

只读列表查询示例：

```text
http://127.0.0.1:8001/tasks
http://127.0.0.1:8001/tasks?limit=3
http://127.0.0.1:8001/tasks?status=api_error&limit=3
```

列表返回`{"items": [...]}`，每项包含上述五字段。`limit`默认2、范围1至20；`status`可不传，不传时查全部，传入时按状态精确匹配。当前接受任意状态字符串，找不到匹配记录返回HTTP200及空列表。`status=`为空字符串，与未传状态不同。

单条查询编号不存在时返回HTTP404。查询不会重新调用模型。

`GET /task-stats`返回`{"total": 总任务数, "items": [{"status": 状态, "count": 数量}]}`，按状态升序排列。统计覆盖全部已保存任务，不受任务列表的`limit`影响。没有任务时返回`{"total": 0, "items": []}`，不存在的状态不会自动补零。`answered`数量仅表示相应业务状态的记录数，不代表答案准确率。

## 数据保存与验证范围

SQLite文件为`data/api_tasks.db`，启动时建表，任务结束后插入并提交最终结果。正常重启保留已提交记录；此数据库及伴随文件已被Git忽略。数据库不保存文档向量，也不会随仓库分发；迁移或容器化时需安排数据文件的保存位置。

当前已有成功任务保存与独立读取、正常重启后查询、模拟失败记录、列表数量校验及可选状态查询的局部验证。尚未验证干净环境重建、真实模型失败经POST的端到端路径、异常停机或并发行为。其他系统上的依赖与本地模型加载仍需单独验证。

任务状态统计已在当前含数据的本地服务上验证HTTP200、总数与各状态数量之和一致。空表及新增状态场景尚未单独运行验证。

另已通过进程内HTTP请求模拟Agent结果及保存函数，核对POST保存成功的200五字段响应，以及保存函数抛出SQLite错误后的500错误响应。该验证未调用真实模型或写入数据库，不代表真实数据库故障的端到端验证。
