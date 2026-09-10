"""Streamlit interface for the Domain-Specific PDF RAG Chatbot."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from document_loader import PDFValidationError, extract_pages
from rag_pipeline import FALLBACK_RESPONSE, RAGPipeline


PROJECT_DIR = Path(__file__).resolve().parent
SAVED_INDEX_DIR = PROJECT_DIR / "saved_index"
MAX_FILE_SIZE_MB = 10


def reset_chat() -> None:
    """Remove only chat messages; retain the processed document collection."""
    st.session_state.messages = []


def clear_everything() -> None:
    """Reset the in-memory collection and remove an optional saved index."""
    st.session_state.pipeline = None
    st.session_state.messages = []
    if SAVED_INDEX_DIR.exists():
        shutil.rmtree(SAVED_INDEX_DIR)


def show_sources(sources) -> None:
    """Render source citations without exposing entire documents by default."""
    if not sources:
        return
    with st.expander("Sources used", expanded=False):
        for source in sources:
            name = source.chunk.metadata["document_name"]
            page = source.chunk.metadata["page_number"]
            st.markdown(f"- **{name}**, page {page} - relevance: {source.score:.2f}")
            excerpt = source.chunk.text[:350].strip()
            st.caption(excerpt + ("..." if len(source.chunk.text) > 350 else ""))


def main() -> None:
    load_dotenv(PROJECT_DIR / ".env")
    st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📚", layout="wide")

    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.title("📚 Domain-Specific PDF RAG Chatbot")
    st.caption("Ask questions about your PDFs. Answers are grounded in retrieved pages and include source citations.")
    st.info("For medical, legal, financial, or other high-stakes information, verify answers against the original document.")

    with st.sidebar:
        st.header("Document collection")
        uploaded_files = st.file_uploader(
            "Upload one or more PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help=f"Each PDF must be no larger than {MAX_FILE_SIZE_MB} MB.",
        )
        if uploaded_files:
            st.caption("Selected files")
            for uploaded_file in uploaded_files:
                st.write(f"- {uploaded_file.name} ({uploaded_file.size / 1_048_576:.1f} MB)")

        save_index = st.checkbox("Save this index locally", value=False)
        if st.button("Process documents", type="primary", use_container_width=True):
            if not uploaded_files:
                st.warning("Upload at least one PDF before processing.")
            else:
                try:
                    with st.spinner("Reading pages, making chunks, and building the FAISS index..."):
                        pages = extract_pages(uploaded_files, max_file_size_mb=MAX_FILE_SIZE_MB)
                        pipeline = RAGPipeline()
                        stats = pipeline.build_index(pages)
                        if save_index:
                            pipeline.save_index(SAVED_INDEX_DIR)
                        st.session_state.pipeline = pipeline
                        st.session_state.messages = []
                    st.success(
                        f"Ready: {stats['documents']} document(s), {stats['pages']} text page(s), "
                        f"and {stats['chunks']} chunks."
                    )
                except (PDFValidationError, ValueError) as exc:
                    st.error(str(exc))
                except ImportError as exc:
                    st.error(f"A required package is missing: {exc}. Run: pip install -r requirements.txt")
                except Exception as exc:
                    st.error(f"Unable to process the files: {exc}")

        col1, col2 = st.columns(2)
        col1.button("Clear chat", use_container_width=True, on_click=reset_chat)
        col2.button("Clear docs", use_container_width=True, on_click=clear_everything)

        st.divider()
        st.header("LLM settings")
        api_key = st.text_input(
            "Groq API key",
            value=os.getenv("GROQ_API_KEY", ""),
            type="password",
            help="Used only for this session. Put it in .env for local development; never commit it.",
        )
        model = st.text_input("Groq model", value=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"))
        st.caption("Without an API key, documents can be indexed but the chatbot will not generate an answer.")

    if st.session_state.pipeline is None:
        st.markdown("### Getting started")
        st.markdown(
            "1. Upload PDFs in the sidebar.\n"
            "2. Select **Process documents**.\n"
            "3. Add a Groq API key, then ask a question below."
        )
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                show_sources(message.get("sources", []))

    question = st.chat_input("Ask a question about the uploaded PDFs")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant pages..."):
            result = st.session_state.pipeline.answer(question, api_key=api_key, model=model)
        st.markdown(result.answer)
        show_sources(result.sources)
        if result.answer == FALLBACK_RESPONSE:
            st.caption("Try a more specific question or process documents that contain the information.")
    st.session_state.messages.append(
        {"role": "assistant", "content": result.answer, "sources": result.sources}
    )


if __name__ == "__main__":
    main()
