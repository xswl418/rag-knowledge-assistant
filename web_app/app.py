from zipfile import BadZipFile
from docx.opc.exceptions import PackageNotFoundError
from pypdf.errors import PdfReadError
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

# Streamlit may add only the entrypoint directory to its import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_assistant.api_client import create_api_client
from rag_assistant.agent import run_agent, AGENT_LOG_PATH
from rag_assistant.conversation import build_history_messages
from rag_assistant import rag
import streamlit as st
import hashlib
import os
import time

ASSETS_DIR = Path(__file__).parent / "assets"
READING_WIDTH = 820

st.set_page_config(
    page_title="文档问答",
    page_icon=":material/auto_stories:",
    layout="wide",
    initial_sidebar_state="auto",
)
st.logo(
    str(ASSETS_DIR / "logo.svg"),
    size="large",
    icon_image=":material/auto_stories:",
)
st.html("<style>" + (ASSETS_DIR / "styles.css").read_text(encoding="utf-8") + "</style>")


@st.cache_resource(max_entries=1, show_spinner="正在准备检索……")
def load_embedding_model(model_name):
    import_start = time.perf_counter()
    from sentence_transformers import SentenceTransformer
    import_seconds = time.perf_counter() - import_start
    print(f"向量库导入耗时：{import_seconds:.3f} 秒")

    print("开始加载向量模型……")

    model_start = time.perf_counter()
    allow_download = os.environ.get("EMBEDDING_ALLOW_DOWNLOAD", "false").lower() == "true"
    model = SentenceTransformer(
        model_name,
        local_files_only=not allow_download,
    )
    model_seconds = time.perf_counter() - model_start
    print(f"向量模型初始化耗时：{model_seconds:.3f} 秒")
    return model

@st.cache_data(
    scope="session",
    max_entries=1,
    show_spinner="正在处理文档……",
)

def create_document_vectors(chunks, model_name):
    print("开始计算文档向量……")

    embedding_model = load_embedding_model(model_name)
    chunk_vectors = rag.create_embeddings(chunks, embedding_model)

    return chunk_vectors

@st.cache_data(
    scope="session",
    max_entries=1,
    show_spinner="正在读取文档……",
)
def parse_uploaded_document(file_bytes, file_name, max_chunk_size, overlap):
    print("开始读取并切分文档……")

    with TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / Path(file_name).name
        file_path.write_bytes(file_bytes)

        chunks = rag.load_document_chunks(
            file_path,
            max_chunk_size=max_chunk_size,
            overlap=overlap,
        )

    return chunks


def format_file_size(size):
    if size >= 1024 * 1024:
        return f"{size / 1024 / 1024:.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size} B"


def describe_document(chunks):
    """Tell the reader how much text was actually read, in plain words."""
    if any("page_number" in chunk for chunk in chunks):
        pages = {chunk["page_number"] for chunk in chunks}
        return f"已读取 {len(pages)} 页文字"
    groups = {chunk["paragraph_id"] for chunk in chunks}
    return f"已读取 {len(groups)} 段文字"


def chunk_location(chunk):
    if "page_number" in chunk:
        return f"第 {chunk['page_number']} 页"
    return f"第 {chunk['paragraph_id']} 段"


def render_welcome(document_ready=False):
    """Show the next useful action before a conversation begins."""
    with st.container(border=True, gap="small", key="welcome"):
        if document_ready:
            st.header("文档已就绪，问一个具体的问题。", anchor=False)
            st.markdown(
                "可以问某项规定的条件、某个概念的定义，或者某个具体数字。"
                "也可以要求概括全文，或者按指定字段提取信息，例如：提取借阅上限、续借次数和联系电话。"
                "回答下方可以展开本次检索到的资料，回到原文核对。"
            )
        else:
            st.header("把文档放进来，把问题说出来。", anchor=False)
            st.markdown(
                "支持 TXT、DOCX 和带文字层的 PDF，一次一份。"
                "上传后围绕这份文档提问，也可以沿着上一轮回答继续追问。"
            )
        hints = st.columns(3, gap="small")
        features = (
            (":material/search:", "查找信息", "定义、条件、数字和具体条款。"),
            (":material/fact_check:", "提取字段", "按指定字段整理文档信息。"),
            (":material/forum:", "接着追问", "沿着上一轮回答继续问下去。"),
        )
        for hint, (icon, title, description) in zip(hints, features):
            with hint:
                st.markdown(f"{icon} **{title}**")
                st.caption(description)


def format_extraction_sources(sources):
    source_texts = []
    for source in sources:
        location = ""
        for key, label in (
            ("page_number", "页"),
            ("paragraph_id", "分组"),
            ("chunk_id", "块"),
        ):
            number = source.get(key)
            if number is not None:
                location = f"第 {number} 页" if key == "page_number" else f"{label} {number}"
                break
        filename = source.get("source", "未知文件")
        source_texts.append(f"{filename}，{location}" if location else filename)
    return "\n".join(source_texts)


def render_extraction_result(extraction_data, key_prefix):
    """Keep each candidate value paired with its quotations and real locations."""
    items = extraction_data.get("items", [])

    if not items:
        st.info("没有提取到任何字段信息。", icon=":material/info:")
        return

    st.caption(f"共提取 {len(items)} 个字段")

    for idx, item in enumerate(items):
        field = item.get("field", "")
        options = item.get("options")
        if options is None:
            options = [] if item.get("value") is None else [{
                "value": item["value"],
                "evidences": [{
                    "text": item.get("evidence"),
                    "sources": item.get("sources", []),
                }],
            }]

        with st.container(border=True, gap="small", key=f"extraction-item-{key_prefix}-{idx}"):
            st.markdown(f"**{field}**")

            if not options:
                st.text("未找到")
                continue

            if len(options) > 1:
                st.badge("多项结果，待核对", icon=":material/compare_arrows:", color="orange")

            for option_index, option in enumerate(options, start=1):
                if option_index > 1:
                    st.divider()
                st.caption(f"候选 {option_index}" if len(options) > 1 else "提取值")
                st.text(option["value"])
                for evidence_index, evidence in enumerate(option["evidences"], start=1):
                    st.caption(f"原文依据 {evidence_index}")
                    st.text(evidence.get("text") or "—")
                    st.caption("来源")
                    st.text(format_extraction_sources(evidence.get("sources", [])) or "—")


def render_retrieval_sources(record, key_prefix):
    """Display actual retrieved text, without presenting it as verified citations."""
    steps = record.get("retrieval_steps") or []
    result_count = sum(len(step["results"]) for step in steps)
    if result_count == 0:
        return

    with st.expander(
        f"本次检索到 {result_count} 条资料",
        icon=":material/library_books:",
    ):
        st.caption(
            "这些是系统从文档中找出的片段，可用来核对回答；"
            "不代表每一条都被回答采用。"
        )
        source_index = 0
        for step in steps:
            if len(steps) > 1:
                st.caption("检索问题")
                st.text(step["question"])
            for result in step["results"]:
                chunk = result["chunk"]
                source_index += 1
                with st.container(gap="xsmall", key=f"source-{key_prefix}-{source_index}"):
                    with st.container(horizontal=True, gap="small", vertical_alignment="center"):
                        st.badge(chunk_location(chunk), color="orange")
                        st.caption(chunk["source"], width="content")
                        if "score" in result:
                            st.caption(f"相关度 {result['score']:.2f}", width="content")
                    st.text(chunk["content"])


def render_answer(record, key_prefix):
    if record["status"] == "answered":
        # 区分提取结果和普通回答
        extraction_result = record.get("extraction_result")
        if extraction_result is not None:
            render_extraction_result(extraction_result, key_prefix)
        else:
            st.markdown(record["answer"])
    elif record["status"] == "extraction_error":
        error_msg = record.get("error_message")
        if error_msg:
            st.error(error_msg, icon=":material/error:")
        else:
            st.error("提取失败，请稍后重试。", icon=":material/error:")
    elif record["status"] == "retrieval_error":
        error_msg = record.get("error_message")
        if error_msg:
            st.error(error_msg, icon=":material/error:")
        else:
            st.error("检索准备失败，请稍后重试。", icon=":material/error:")
    elif record["status"] == "no_results":
        st.warning(
            "没有在文档中找到相关内容。换个说法，或者把问题问得更具体一些。",
            icon=":material/search_off:",
        )
    elif record["status"] == "no_answer":
        st.warning("这次没有生成出回答，请再提交一次。", icon=":material/info:")
    elif record["status"] == "api_error":
        st.error("暂时连不上回答服务，请稍后重试。", icon=":material/cloud_off:")
    elif record["status"] == "summary_error":
        st.warning(
            record.get("error_message") or "这次没能完成文档概括，请稍后重试。",
            icon=":material/info:",
        )
    elif record["status"] == "max_calls":
        st.warning(
            "这个问题的处理步骤过多，没能完成。把问题拆小一些再试。",
            icon=":material/info:",
        )
    else:
        st.warning("这次没能完成回答，请再提交一次。", icon=":material/info:")
    render_retrieval_sources(record, key_prefix)


def render_user_message(question, key_prefix):
    with st.container(key=f"user-{key_prefix}"):
        with st.chat_message("user", avatar=":material/person:"):
            st.text(question)


def render_assistant_message(record, key_prefix):
    with st.container(key=f"assistant-{key_prefix}"):
        with st.chat_message("assistant", avatar=":material/auto_stories:"):
            preface = (
                record.get("extraction_preface")
                or record.get("summary_preface")
                or record.get("retrieval_preface")
            )
            if preface:
                st.markdown(preface)
            render_answer(record, key_prefix)


def render_question_form(document_ready, placeholder=None):
    """Keep the existing form submission behavior in a bottom composer."""
    if placeholder is None:
        placeholder = (
            "输入问题，或接着上一轮追问"
            if document_ready
            else "先上传一份文档，再开始提问"
        )
    with st.bottom:
        with st.container(horizontal_alignment="center"):
            with st.container(width=READING_WIDTH, gap="xsmall", key="composer"):
                with st.form("question_form", border=False, clear_on_submit=True):
                    with st.container(
                        horizontal=True,
                        vertical_alignment="center",
                        gap="xsmall",
                    ):
                        question = st.text_input(
                            "你的问题",
                            label_visibility="collapsed",
                            placeholder=placeholder,
                            key="question_input",
                            disabled=not document_ready,
                        )
                        submitted = st.form_submit_button(
                            "提问",
                            icon=":material/arrow_upward:",
                            type="primary",
                            disabled=not document_ready,
                        )
                st.caption("回答依据当前文档生成，重要信息请核对原文。")
    return question, submitted


def stop_with_document_error(message):
    """Show a parse failure in both the document card and the main area."""
    with document_card:
        st.badge("无法读取", icon=":material/error:", color="red")
    with status_bar:
        st.error(message, icon=":material/error:")
    render_question_form(
        document_ready=False,
        placeholder="换一份可以读取的文档后再提问",
    )
    st.stop()


if "chat_records" not in st.session_state:
    st.session_state["chat_records"] = []

if "current_document_id" not in st.session_state:
    st.session_state["current_document_id"] = None

with st.sidebar:
    st.subheader("当前文档", anchor=False)
    document_card = st.container(gap="small")
    uploaded_file = st.file_uploader(
        "上传或更换文档",
        type=["txt", "docx", "pdf"],
        accept_multiple_files=False,
        key="document_upload",
        help="支持 TXT、DOCX 和带文字层的 PDF，一次一份。",
    )
    document_tools = st.container(gap="small")
    st.caption("更换或移除文档后，对话会重新开始。")
    with st.expander("使用提示", icon=":material/help:"):
        st.markdown(
            "- 一次围绕一份文档提问。\n"
            "- 可以接着上一轮追问；清空对话后会重新开始。\n"
            "- PDF 需要包含可提取的文字，扫描件暂不支持。\n"
            "- TXT 请使用 UTF-8 编码。"
        )

with st.container(horizontal_alignment="center"):
    workspace = st.container(width=READING_WIDTH, gap="medium")

with workspace:
    status_bar = st.container(gap="small", key="status-bar")
    conversation_area = st.container(gap="medium")

current_document_id = None

if uploaded_file is not None:
    file_hash = hashlib.sha256(
        uploaded_file.getvalue()
    ).hexdigest()

    current_document_id = (uploaded_file.name, file_hash)

if current_document_id != st.session_state["current_document_id"]:
    st.session_state["chat_records"] = []
    st.session_state["current_document_id"] = current_document_id
    st.session_state["question_input"] = ""

if uploaded_file is None:
    with status_bar:
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.badge("还没有文档", icon=":material/upload_file:", color="gray")
            st.caption("在左侧文档面板上传一份 TXT、DOCX 或 PDF。", width="content")
    with conversation_area:
        render_welcome()
    render_question_form(document_ready=False)
    st.stop()

with document_card:
    with st.container(border=True, gap="xsmall", key="document-card"):
        st.text(uploaded_file.name)
        st.caption(
            f"{Path(uploaded_file.name).suffix[1:].upper()} 文件，"
            f"{format_file_size(uploaded_file.size)}"
        )

if uploaded_file.size == 0:
    stop_with_document_error("这份文档是空的。请添加内容后重新上传。")

chunks = None
error_message = None

with status_bar:
    try:
        chunks = parse_uploaded_document(
            file_bytes=uploaded_file.getvalue(),
            file_name=uploaded_file.name,
            max_chunk_size=rag.MAX_CHUNK_SIZE,
            overlap=rag.CHUNK_OVERLAP,
        )

    except UnicodeDecodeError:
        error_message = "无法读取文本编码。请把 TXT 另存为 UTF-8 编码后重新上传。"

    except (PackageNotFoundError, BadZipFile):
        error_message = "无法识别这个 Word 文件。请上传有效的 DOCX 文档。"

    except PdfReadError:
        error_message = "PDF 读取失败。请上传有效且无需密码的 PDF 文件。"

    except OSError:
        error_message = "文件保存或读取失败，请重新上传。"

if error_message is not None:
    stop_with_document_error(error_message)

if not chunks:
    stop_with_document_error(
        "没有读取到可用文字。请检查文件内容；扫描件暂不支持识别。"
    )

with document_card:
    st.caption(describe_document(chunks))

with document_tools:
    with st.expander("查看文档内容", icon=":material/article:"):
        with st.container(height=320, border=False, gap="small"):
            for chunk in chunks:
                st.caption(chunk_location(chunk))
                st.text(chunk["content"])

with status_bar:
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.badge("文档已读取", icon=":material/check_circle:", color="green")
            st.caption(uploaded_file.name, width="content")
        if st.button(
            "清空对话",
            icon=":material/delete_sweep:",
            type="tertiary",
            key="clear_conversation",
        ):
            st.session_state["chat_records"] = []
            st.session_state["question_input"] = ""
            st.toast("对话已清空", icon=":material/check:")

question, submitted = render_question_form(document_ready=True)

with conversation_area:
    for index, record in enumerate(st.session_state["chat_records"]):
        render_user_message(record["question"], index)
        render_assistant_message(record, index)

    if not st.session_state["chat_records"] and not (submitted and question.strip()):
        render_welcome(document_ready=True)

    if submitted:
        question = question.strip()

        if not question:
            st.warning("请输入问题后再提交。", icon=":material/edit:")
        else:
            history_messages = build_history_messages(
                st.session_state["chat_records"],
            )

            pending_index = len(st.session_state["chat_records"])
            render_user_message(question, pending_index)
            with st.container(key=f"assistant-{pending_index}"):
                with st.chat_message("assistant", avatar=":material/auto_stories:"):

                    status_container = st.status("正在判断问题需要哪些资料……", expanded=True)

                    streamed_preface = st.empty()
                    streamed_answer = st.empty()
                    streamed_parts = []
                    tool_preface = []
                    answer_started = [False]

                    def handle_text_piece(text):
                        streamed_parts.append(text)
                        target = streamed_answer if answer_started[0] else streamed_preface
                        target.markdown("".join(streamed_parts))

                    def begin_answer_text():
                        if not answer_started[0]:
                            tool_preface.append("".join(streamed_parts).strip())
                            streamed_parts.clear()
                            answer_started[0] = True

                    def prepare_retrieval_resources():
                        """只在真正需要检索时才准备向量资源"""
                        begin_answer_text()
                        status_container.update(label="正在准备检索资源", state="running")
                        status_container.write("需要检索文档资料")
                        status_container.info(
                            "如果这是服务启动后的首次检索，需要准备本地模型，可能等待较久；"
                            "后续可复用模型。较长文档也需要更多准备时间。",
                            icon=":material/schedule:"
                        )

                        status_container.update(label="正在准备检索模型", state="running")
                        embedding_model = load_embedding_model(rag.EMBEDDING_MODEL_NAME)

                        status_container.update(label="正在准备文档检索资料", state="running")
                        chunk_vectors = create_document_vectors(chunks, rag.EMBEDDING_MODEL_NAME)

                        status_container.update(label="正在检索资料并组织回答", state="running")
                        print(f"检索资源准备完成，向量形状：{chunk_vectors.shape}")
                        return embedding_model, chunk_vectors

                    # 摘要进度跟踪
                    summary_progress_container = st.empty()
                    summary_hint_shown = [False]  # 使用列表来允许闭包内修改
                    progress_bar = [None]  # 存储进度条引用

                    def handle_summary_progress(event):
                        """接收摘要进度事件并更新UI"""
                        stage = event.get("stage", "")
                        completed = event.get("completed", 0)
                        total = event.get("total", 1)
                        message = event.get("message", "")

                        if stage in ("batch_started", "merge_started"):
                            begin_answer_text()

                        # 首次摘要事件时显示提示
                        if not summary_hint_shown[0]:
                            summary_hint_shown[0] = True
                            status_container.info(
                                "短文直接概括；长文按批次概括后汇总，可能需要较长时间。进度按处理步骤计算。",
                                icon=":material/info:"
                            )

                        # 计算进度值
                        progress_value = completed / total if total > 0 else 0.0

                        # 更新或创建进度条
                        with summary_progress_container:
                            progress_text = f"{message} ({completed}/{total})"
                            progress_bar[0] = st.progress(progress_value, text=progress_text)

                        # 更新状态容器标签
                        if stage == "batch_started":
                            status_container.update(label="正在概括文档", state="running")
                        elif stage == "batch_completed":
                            status_container.update(label="正在概括文档", state="running")
                        elif stage == "merge_started":
                            status_container.update(label="正在汇总概括结果", state="running")
                        elif stage == "summary_completed":
                            status_container.update(label="正在组织回答", state="running")

                    extraction_progress_container = st.empty()

                    def handle_extraction_progress(event):
                        begin_answer_text()
                        completed = event["completed"]
                        total = event["total"]
                        message = event["message"]
                        with extraction_progress_container:
                            st.progress(
                                completed / total if total else 0.0,
                                text=f"{message} ({completed}/{total})",
                            )
                        label = (
                            "正在合并提取结果"
                            if event["stage"] == "batch_completed" and completed == total
                            else message
                        )
                        status_container.update(label=label, state="running")

                    record = None
                    try:
                        with create_api_client() as client:
                            record = run_agent(
                                question=question,
                                client=client,
                                knowledge_chunks=chunks,
                                chunk_vectors=None,
                                embedding_model=None,
                                history_messages=history_messages,
                                get_retrieval_resources=prepare_retrieval_resources,
                                on_summary_progress=handle_summary_progress,
                                on_text=handle_text_piece,
                                on_extraction_progress=handle_extraction_progress,
                            )
                    finally:
                        if record is None or record["status"] != "answered":
                            streamed_preface.empty()
                            streamed_answer.empty()
                        summary_progress_container.empty()
                        extraction_progress_container.empty()

                    # 根据状态更新状态容器
                    if record["status"] == "answered":
                        status_container.update(label="回答完成", state="complete", expanded=False)
                    elif record["status"] == "no_results":
                        status_container.update(label="未找到相关资料", state="complete", expanded=False)
                    elif record["status"] in ["retrieval_error", "api_error", "extraction_error", "summary_error", "max_calls", "no_answer", "unsupported_tool_calls"]:
                        status_container.update(label="处理遇到错误", state="error", expanded=False)
                    else:
                        status_container.update(label="处理未完成", state="error", expanded=False)

                    if record["status"] == "answered":
                        preface = "".join(tool_preface)
                        if preface:
                            if record.get("extraction_result") is not None:
                                preface_key = "extraction_preface"
                            else:
                                preface_key = "summary_preface" if record.get("summary_input") else "retrieval_preface"
                            record[preface_key] = preface
                        else:
                            streamed_preface.empty()
                        if record.get("extraction_result") is not None:
                            streamed_answer.empty()
                            render_answer(record, pending_index)
                        else:
                            streamed_answer.markdown(record["answer"])
                            render_retrieval_sources(record, pending_index)
                    else:
                        streamed_preface.empty()
                        streamed_answer.empty()
                        render_answer(record, pending_index)

            record["source"] = uploaded_file.name
            st.session_state["chat_records"].append(record)
            rag.save_record(record, AGENT_LOG_PATH)
