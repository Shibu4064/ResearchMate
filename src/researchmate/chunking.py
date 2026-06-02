from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .document_loader import LoadedDocument


@dataclass
class TextChunk:
    text: str
    source: str
    doc_type: str
    page: int | None
    chunk_id: int


def split_text(text: str, chunk_size: int = 850, overlap: int = 140) -> list[str]:
    """Simple paragraph-aware chunker with character overlap."""
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 1 <= chunk_size:
            current = f"{current}\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            if len(para) > chunk_size:
                start = 0
                while start < len(para):
                    end = min(start + chunk_size, len(para))
                    chunks.append(para[start:end])
                    start = max(end - overlap, end)
            else:
                current = para

    if current:
        chunks.append(current)

    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped: list[str] = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            overlapped.append(chunk)
        else:
            prefix = chunks[i - 1][-overlap:]
            overlapped.append(f"{prefix}\n{chunk}")
    return overlapped


def chunk_documents(documents: Iterable[LoadedDocument], chunk_size: int, overlap: int) -> list[TextChunk]:
    output: list[TextChunk] = []
    chunk_counter = 0
    for doc in documents:
        for chunk_text in split_text(doc.text, chunk_size=chunk_size, overlap=overlap):
            output.append(
                TextChunk(
                    text=chunk_text,
                    source=doc.source,
                    doc_type=doc.doc_type,
                    page=doc.page,
                    chunk_id=chunk_counter,
                )
            )
            chunk_counter += 1
    return output
