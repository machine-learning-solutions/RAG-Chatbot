import os
from typing import Any

import httpx
import streamlit as st

from rtl import page_css, render_bidi_text, t

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
HEALTH_TIMEOUT = 5.0
API_TIMEOUT = 60.0
CHAT_TIMEOUT = 300.0

if "language" not in st.session_state:
    st.session_state.language = "ar"


def current_lang() -> str:
    return st.session_state.language


def is_portfolio_mode() -> bool:
    embed = st.query_params.get("embed", "")
    portfolio = st.query_params.get("portfolio", "")
    env_flag = os.getenv("PORTFOLIO_MODE", "").lower() == "true"
    return embed == "true" or portfolio == "true" or env_flag


portfolio_mode = is_portfolio_mode()
lang = current_lang()

st.set_page_config(
    page_title="Portfolio Assistant" if portfolio_mode else t(lang, "page_title"),
    page_icon="💼" if portfolio_mode else "📚",
    layout="wide",
    initial_sidebar_state="collapsed" if portfolio_mode else "auto",
)

st.markdown(page_css(lang, portfolio=portfolio_mode), unsafe_allow_html=True)


def api_get(path: str, timeout: float = API_TIMEOUT) -> dict[str, Any]:
    with httpx.Client(timeout=timeout) as client:
        response = client.get(f"{BACKEND_URL}{path}")
        response.raise_for_status()
        return response.json()


def _parse_sse_buffer(buffer: str) -> tuple[list[dict[str, str]], str]:
    events: list[dict[str, str]] = []
    parts = buffer.split("\n\n")
    rest = parts.pop() if parts else buffer
    for part in parts:
        if not part.strip():
            continue
        event = "message"
        data = ""
        for line in part.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data += line[5:].strip()
        events.append({"event": event, "data": data})
    return events, rest


def api_post_stream(
    path: str,
    json: dict | None = None,
    timeout: float = CHAT_TIMEOUT,
) -> dict[str, Any]:
    import json as json_lib

    with httpx.Client(timeout=timeout) as client:
        with client.stream(
            "POST",
            f"{BACKEND_URL}{path}",
            json={**(json or {}), "stream": True},
            headers={"Accept": "text/event-stream"},
            timeout=timeout,
        ) as response:
            response.raise_for_status()
            buffer = ""
            for chunk in response.iter_bytes():
                buffer += chunk.decode()
                events, buffer = _parse_sse_buffer(buffer)
                for evt in events:
                    if evt["event"] == "error":
                        detail = json_lib.loads(evt["data"]).get("detail", "Chat failed")
                        raise RuntimeError(detail)
                    if evt["event"] == "result":
                        return json_lib.loads(evt["data"])
    raise RuntimeError("Chat ended without a result")


def api_post(
    path: str,
    json: dict | None = None,
    files: dict | None = None,
    timeout: float = CHAT_TIMEOUT,
) -> dict[str, Any]:
    with httpx.Client(timeout=timeout) as client:
        if files:
            response = client.post(f"{BACKEND_URL}{path}", files=files, timeout=timeout)
        else:
            response = client.post(f"{BACKEND_URL}{path}", json=json, timeout=timeout)
        response.raise_for_status()
        return response.json()


def api_delete(path: str) -> dict[str, Any]:
    with httpx.Client(timeout=API_TIMEOUT) as client:
        response = client.delete(f"{BACKEND_URL}{path}")
        response.raise_for_status()
        return response.json()


def init_portfolio_defaults() -> None:
    st.session_state.setdefault("top_k", 5)
    st.session_state.setdefault("use_reranker", True)
    st.session_state.setdefault("use_hybrid", True)
    st.session_state.setdefault("document_id", None)


def set_language(code: str) -> None:
    if st.session_state.language == code:
        return

    st.session_state.language = code
    welcome_key = "portfolio_welcome" if portfolio_mode else "welcome"
    messages = st.session_state.get("messages", [])
    has_user_messages = any(message.get("role") == "user" for message in messages)

    if not has_user_messages:
        st.session_state.messages = [
            {"role": "assistant", "content": t(code, welcome_key), "sources": []}
        ]

    st.rerun()


def render_language_switcher(*, compact: bool = False) -> None:
    lang = current_lang()
    if compact:
        col1, col2, _ = st.columns([1, 1, 4])
    else:
        col1, col2 = st.sidebar.columns(2)

    with col1:
        if st.button(
            t("en", "lang_en"),
            use_container_width=True,
            type="primary" if lang == "en" else "secondary",
            key="lang_en",
        ):
            set_language("en")
    with col2:
        if st.button(
            t("ar", "lang_ar"),
            use_container_width=True,
            type="primary" if lang == "ar" else "secondary",
            key="lang_ar",
        ):
            set_language("ar")


def render_chat() -> None:
    lang = current_lang()

    if not portfolio_mode:
        st.title(t(lang, "title"))
        st.caption(t(lang, "caption"))
    else:
        render_language_switcher(compact=True)

    welcome_key = "portfolio_welcome" if portfolio_mode else "welcome"
    chat_input_key = "portfolio_chat_input" if portfolio_mode else "chat_input"

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": t(lang, welcome_key), "sources": []}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            render_bidi_text(message["content"])

    if prompt := st.chat_input(t(lang, chat_input_key)):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            render_bidi_text(prompt)

        with st.chat_message("assistant"):
            with st.spinner(t(lang, "searching")):
                try:
                    payload = {
                        "question": prompt,
                        "language": st.session_state.language,
                        "document_id": st.session_state.get("document_id"),
                        "top_k": st.session_state.get("top_k", 5),
                        "use_reranker": st.session_state.get("use_reranker", True),
                        "use_hybrid": st.session_state.get("use_hybrid", True),
                        "portfolio_fast": portfolio_mode,
                    }
                    if portfolio_mode:
                        result = api_post_stream("/api/chat", json=payload)
                    else:
                        result = api_post("/api/chat", json=payload)
                    answer = result.get("answer", "No response")

                    render_bidi_text(answer)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": [],
                        }
                    )
                except Exception as exc:
                    error = f"Error: {exc}"
                    st.error(error)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error}
                    )


def render_sidebar() -> str | None:
    st.sidebar.title(t(lang, "sidebar_title"))
    st.sidebar.caption(t(lang, "sidebar_caption"))

    render_language_switcher()
    st.sidebar.divider()

    selected_doc: str | None = None

    with st.sidebar.expander(t(lang, "system_status"), expanded=False):
        try:
            health = api_get("/api/health", timeout=HEALTH_TIMEOUT)
            status = health.get("status", "unknown")
            st.write(f"**Status:** {status.upper()}")

            ollama = health.get("ollama", {})
            if ollama.get("status") == "ok":
                models = ollama.get("models", [])
                st.write(f"**Ollama:** {', '.join(models[:3]) or 'none'}")
            else:
                st.warning("Ollama not ready")

            vector = health.get("vector_store", {})
            st.write(
                f"**TurboVec:** {'indexed' if vector.get('indexed') else 'empty'} "
                f"(bit_width={vector.get('bit_width', 4)})"
            )
        except Exception as exc:
            st.sidebar.error(f"Backend unreachable: {exc}")

    st.sidebar.divider()
    st.sidebar.subheader(t(lang, "upload"))
    st.sidebar.caption(t(lang, "upload_types"))

    uploaded = st.sidebar.file_uploader(
        "Choose file",
        type=["pdf", "docx", "doc", "md", "txt", "json"],
        label_visibility="collapsed",
    )

    if uploaded and st.sidebar.button(t(lang, "ingest"), use_container_width=True):
        with st.sidebar.status("Processing...", expanded=True) as status:
            try:
                files = {
                    "file": (
                        uploaded.name,
                        uploaded.getvalue(),
                        uploaded.type or "application/octet-stream",
                    )
                }
                result = api_post("/api/documents/upload", files=files)
                status.update(
                    label=f"Done: {result['chunks_created']} chunks",
                    state="complete",
                )
            except Exception as exc:
                status.update(label="Failed", state="error")
                st.sidebar.error(str(exc))

    st.sidebar.divider()
    st.sidebar.subheader(t(lang, "filter"))

    try:
        docs_data = api_get("/api/documents", timeout=HEALTH_TIMEOUT)
        documents = docs_data.get("documents", [])
    except Exception:
        documents = []

    if documents:
        options = {t(lang, "all_docs"): None}
        for doc in documents:
            options[f"{doc['filename']} ({doc['chunk_count']} chunks)"] = doc["id"]

        choice = st.sidebar.selectbox("Search scope", list(options.keys()))
        selected_doc = options[choice]

        st.sidebar.divider()
        st.sidebar.subheader(t(lang, "manage"))

        for doc in documents:
            col1, col2 = st.sidebar.columns([3, 1])
            col1.caption(f"📄 {doc['filename']}")
            if col2.button("🗑", key=f"del_{doc['id']}"):
                try:
                    api_delete(f"/api/documents/{doc['id']}")
                    st.rerun()
                except Exception as exc:
                    st.sidebar.error(str(exc))
    else:
        st.sidebar.info(t(lang, "no_docs"))

    st.sidebar.divider()
    top_k = st.sidebar.slider(t(lang, "top_k"), 1, 15, 5)
    use_reranker = st.sidebar.checkbox(t(lang, "reranker"), value=True)
    use_hybrid = st.sidebar.checkbox(t(lang, "hybrid"), value=True)

    st.session_state["top_k"] = top_k
    st.session_state["use_reranker"] = use_reranker
    st.session_state["use_hybrid"] = use_hybrid
    st.session_state["document_id"] = selected_doc

    return selected_doc


if portfolio_mode:
    init_portfolio_defaults()

render_chat()

if not portfolio_mode:
    render_sidebar()
