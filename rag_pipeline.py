"""Chunking, retrieval, and grounded Groq answer generation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from document_loader import PageDocument
from prompt import FALLBACK_RESPONSE, build_messages
from vector_store import Chunk, FaissVectorStore, SearchResult


CHUNK_SIZE = 850
CHUNK_OVERLAP = 125
RETRIEVAL_SCORE_THRESHOLD = 0.28


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    sources: list[SearchResult]


def chunk_pages(
    pages: list[PageDocument], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> list[Chunk]:
    """Create overlapping LangChain chunks without losing page-level provenance."""
    if chunk_size <= chunk_overlap:
        raise ValueError("Chunk size must be greater than overlap.")
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:
        raise ImportError("langchain-text-splitters") from exc
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Chunk] = []
    per_page_counter: Counter[tuple[str, int]] = Counter()
    for page in pages:
        key = (page.metadata["file_hash"], page.metadata["page_number"])
        for text in splitter.split_text(page.text):
            per_page_counter[key] += 1
            metadata = dict(page.metadata)
            metadata["chunk_number"] = per_page_counter[key]
            metadata["chunk_id"] = f"{metadata['file_hash']}-p{metadata['page_number']}-c{metadata['chunk_number']}"
            chunks.append(Chunk(text=text, metadata=metadata))
    if not chunks:
        raise ValueError("The PDFs did not produce any usable text chunks.")
    return chunks


class RAGPipeline:
    """Stateful RAG service used by Streamlit and easily testable in isolation."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        score_threshold: float = RETRIEVAL_SCORE_THRESHOLD,
    ) -> None:
        self.embedding_model = embedding_model
        self.score_threshold = score_threshold
        self.store: FaissVectorStore | None = None
        self.stats: dict[str, int] = {"documents": 0, "pages": 0, "chunks": 0}

    def build_index(self, pages: list[PageDocument]) -> dict[str, int]:
        chunks = chunk_pages(pages)
        self.store = FaissVectorStore.build(chunks, self.embedding_model)
        self.stats = {
            "documents": len({page.metadata["file_hash"] for page in pages}),
            "pages": len(pages),
            "chunks": len(chunks),
        }
        return self.stats

    def save_index(self, directory: Path) -> None:
        if self.store is None:
            raise ValueError("Process documents before saving an index.")
        self.store.save(directory)

    def load_index(self, directory: Path) -> dict[str, int]:
        self.store = FaissVectorStore.load(directory)
        self.stats = {
            "documents": len({chunk.metadata["file_hash"] for chunk in self.store.chunks}),
            "pages": len({(chunk.metadata["file_hash"], chunk.metadata["page_number"]) for chunk in self.store.chunks}),
            "chunks": len(self.store.chunks),
        }
        return self.stats

    def retrieve(self, question: str, top_k: int = 4) -> list[SearchResult]:
        if self.store is None:
            raise ValueError("Process at least one document before asking a question.")
        return self.store.search(question, top_k=top_k)

    def answer(
        self, question: str, api_key: str | None, model: str = "llama-3.1-8b-instant", top_k: int = 4
    ) -> AnswerResult:
        if not question or not question.strip():
            return AnswerResult(answer=FALLBACK_RESPONSE, sources=[])
        retrieved = self.retrieve(question, top_k=top_k)
        supported = [item for item in retrieved if item.score >= self.score_threshold]
        if not supported:
            return AnswerResult(answer=FALLBACK_RESPONSE, sources=retrieved)
        if not api_key or not api_key.strip():
            return AnswerResult(
                answer="Documents were retrieved, but no Groq API key is configured. Add GROQ_API_KEY to .env or the sidebar to generate a grounded answer.",
                sources=supported,
            )
        try:
            from groq import Groq
            client = Groq(api_key=api_key.strip())
            completion = client.chat.completions.create(
                model=model.strip() or "llama-3.1-8b-instant",
                messages=build_messages(question.strip(), supported),
                temperature=0,
                max_tokens=500,
            )
            answer = (completion.choices[0].message.content or "").strip()
            return AnswerResult(answer=answer or FALLBACK_RESPONSE, sources=supported)
        except ImportError:
            return AnswerResult(
                answer="The Groq package is missing. Run: pip install -r requirements.txt", sources=supported
            )
        except Exception as exc:
            return AnswerResult(
                answer=f"The documents were retrieved, but the language model request failed: {exc}",
                sources=supported,
            )
