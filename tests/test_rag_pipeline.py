"""Fast unit tests that do not download an embedding model or call an LLM."""

from document_loader import PageDocument
from rag_pipeline import FALLBACK_RESPONSE, RAGPipeline, chunk_pages
from vector_store import Chunk, SearchResult


def page(text: str, number: int = 1) -> PageDocument:
    return PageDocument(
        text=text,
        metadata={"document_name": "Policy.pdf", "page_number": number, "file_hash": "test-file"},
    )


def test_chunking_keeps_document_and_page_metadata() -> None:
    chunks = chunk_pages([page("Annual leave is 18 days. " * 100)], chunk_size=180, chunk_overlap=30)
    assert len(chunks) > 1
    assert all(chunk.metadata["document_name"] == "Policy.pdf" for chunk in chunks)
    assert all(chunk.metadata["page_number"] == 1 for chunk in chunks)
    assert all("chunk_id" in chunk.metadata for chunk in chunks)


def test_missing_answer_refuses_when_score_is_below_threshold() -> None:
    pipeline = RAGPipeline(score_threshold=0.75)

    class FakeStore:
        def search(self, question, top_k):
            return [SearchResult(Chunk("Leave is 18 days.", page("x").metadata), score=0.20)]

    pipeline.store = FakeStore()
    result = pipeline.answer("Who is the CEO?", api_key="test-key")
    assert result.answer == FALLBACK_RESPONSE


def test_no_api_key_never_invents_an_answer() -> None:
    pipeline = RAGPipeline(score_threshold=0.20)

    class FakeStore:
        def search(self, question, top_k):
            return [SearchResult(Chunk("Leave is 18 days.", page("x").metadata), score=0.91)]

    pipeline.store = FakeStore()
    result = pipeline.answer("How much leave?", api_key="")
    assert "no Groq API key" in result.answer
