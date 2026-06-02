from __future__ import annotations

import hashlib
from dataclasses import asdict
from typing import Iterable

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from .chunking import TextChunk
from .config import settings


class VectorStore:
    """Thin wrapper around ChromaDB with manual SentenceTransformer embeddings.

    Manual embedding keeps the code stable across Chroma embedding-function API changes.
    """

    def __init__(self) -> None:
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        self.client = chromadb.PersistentClient(
            path=str(settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(name=settings.collection_name)

    @staticmethod
    def _make_id(chunk: TextChunk) -> str:
        raw = f"{chunk.source}-{chunk.page}-{chunk.chunk_id}-{chunk.text[:80]}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def reset(self) -> None:
        try:
            self.client.delete_collection(settings.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(name=settings.collection_name)

    def add_chunks(self, chunks: Iterable[TextChunk]) -> int:
        chunks = list(chunks)
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        embeddings = self.embedding_model.encode(texts, normalize_embeddings=True, show_progress_bar=True).tolist()
        ids = [self._make_id(c) for c in chunks]
        metadatas = [
            {
                "source": c.source,
                "doc_type": c.doc_type,
                "page": c.page if c.page is not None else -1,
                "chunk_id": c.chunk_id,
            }
            for c in chunks
        ]

        # Avoid duplicate-ID errors by upserting rather than adding.
        self.collection.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
        return len(chunks)

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        top_k = top_k or settings.top_k
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True).tolist()[0]
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        hits: list[dict] = []
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        for doc, meta, distance in zip(docs, metas, distances):
            hits.append(
                {
                    "text": doc,
                    "source": meta.get("source"),
                    "doc_type": meta.get("doc_type"),
                    "page": None if meta.get("page") == -1 else meta.get("page"),
                    "chunk_id": meta.get("chunk_id"),
                    "distance": float(distance),
                    "similarity": float(1 - distance) if distance is not None else None,
                }
            )
        return hits

    def count(self) -> int:
        return self.collection.count()
