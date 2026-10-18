import os
from pathlib import Path

import streamlit as st

from docuchat import config
from docuchat.loader import load_pdf
from docuchat.qa import DocuChat
from docuchat.store import index_documents, list_sources, remove_source

st.set_page_config(page_title="DocuChat", page_icon="📄", layout="wide")


@st.cache_resource
def load_bot(api_key: str) -> DocuChat:
    # api_key is only here so the cache resets when the key changes
    return DocuChat()


def save_upload(upload) -> Path:
    config.DATA_DIR.mkdir(exist_ok=True)
    path = config.DATA_DIR / upload.name
    path.write_bytes(upload.getbuffer())
    return path


def show_sources(sources: list[dict]):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**[{s['n']}] {s['source']} — page {s['page']}**")
            st.caption(s["text"][:500] + ("..." if len(s["text"]) > 500 else ""))


# ---- sidebar: API key + documents ----
with st.sidebar:
    st.header("Setup")
    api_key = os.getenv("OPENAI_API_KEY") or st.text_input("OpenAI API key", type="password")
    if not api_key:
        st.info("Add your OpenAI key here or in a .env file to get started.")
        st.stop()
    os.environ["OPENAI_API_KEY"] = api_key

    bot = load_bot(api_key)
    bot.k = st.slider("Chunks to retrieve (top-k)", 2, 10, config.TOP_K)

    st.header("Documents")
    uploads = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if uploads and st.button("Index uploaded files", type="primary"):
        for upload in uploads:
            with st.spinner(f"Indexing {upload.name}..."):
                pages = load_pdf(save_upload(upload))
                n = index_documents(bot.store, pages)
            st.success(f"{upload.name}: {n} chunks")

    sources = list_sources(bot.store)
    if not sources:
        st.caption("Nothing indexed yet.")
    for name, count in sources.items():
        col1, col2 = st.columns([4, 1])
        col1.write(f"{name}  \n`{count} chunks`")
        if col2.button("✕", key=f"del-{name}", help="Remove from index"):
            remove_source(bot.store, name)
            st.rerun()


# ---- main: chat ----
st.title("📄 DocuChat")
st.caption("Ask questions about your PDFs. Answers come only from the documents, with page citations.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        show_sources(msg.get("sources", []))

if question := st.chat_input("Ask something about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not sources:
            st.warning("Upload and index at least one PDF first.")
            st.stop()

        with st.spinner("Searching your documents..."):
            answer = bot.ask(question)

        cited = [
            {"n": n, "source": d.metadata["source"], "page": d.metadata["page"], "text": d.page_content}
            for n, d in answer.cited.items()
        ]
        st.markdown(answer.text)
        show_sources(cited)

    st.session_state.messages.append({"role": "assistant", "content": answer.text, "sources": cited})
