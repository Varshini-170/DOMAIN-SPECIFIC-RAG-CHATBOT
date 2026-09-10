"""Sentence-Transformer embeddings and a small, persistent FAISS store."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Chunk:
    text: str
    metadata: dict


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


class FaissVectorStore:
    """Cosine-similarity retrieval over normalised MiniLM embeddings."""

    def __init__(self, index, chunks: list[Chunk], embedding_model: str) -> None:
        self.index = index
        self.chunks = chunks
        self.embedding_model = embedding_model
        self._embedder = None

    @property
    def embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError("sentence-transformers") from exc
            self._embedder = SentenceTransformer(self.embedding_model)
        return self._embedder

    @classmethod
    def build(cls, chunks: Iterable[Chunk], embedding_model: str = "all-MiniLM-L6-v2") -> "FaissVectorStore":
        chunks = list(chunks)
        if not chunks:
            raise ValueError("Cannot build a vector store without text chunks.")
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("faiss-cpu") from exc
        instance = cls(index=None, chunks=chunks, embedding_model=embedding_model)
        vectors = instance.embedder.encode(
            [chunk.text for chunk in chunks], normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        instance.index = index
        return instance

    def search(self, question: str, top_k: int = 4) -> list[SearchResult]:
        if not question.strip():
            return []
        limit = min(max(top_k, 1), len(self.chunks))
        query = self.embedder.encode([question], normalize_embeddings=True, show_progress_bar=False)
        scores, indexes = self.index.search(query.astype("float32"), limit)
        return [
            SearchResult(chunk=self.chunks[index], score=float(score))
            for score, index in zip(scores[0], indexes[0])
            if index >= 0
        ]

    def save(self, directory: Path) -> None:
        """Persist only FAISS vectors and JSON metadata; no unsafe pickle loading."""
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("faiss-cpu") from exc
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        (directory / "chunks.json").write_text(
            json.dumps([asdict(chunk) for chunk in self.chunks], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (directory / "store_config.json").write_text(
            json.dumps({"embedding_model": self.embedding_model}, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> "FaissVectorStore":
        """Load a locally saved collection created by :meth:`save`."""
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("faiss-cpu") from exc
        config = json.loads((directory / "store_config.json").read_text(encoding="utf-8"))
        raw_chunks = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        chunks = [Chunk(**item) for item in raw_chunks]
        index = faiss.read_index(str(directory / "index.faiss"))
        if index.ntotal != len(chunks):
            raise ValueError("Saved index and chunk metadata do not match.")
        return cls(index=index, chunks=chunks, embedding_model=config["embedding_model"])
