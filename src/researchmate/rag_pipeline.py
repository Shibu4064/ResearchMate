from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

from .chunking import chunk_documents
from .config import settings
from .document_loader import SUPPORTED_EXTENSIONS, load_many
from .llm import OllamaClient, build_cv_match_prompt, build_rag_prompt
from .vector_store import VectorStore


class RAGPipeline:
    def __init__(self) -> None:
        self.store = VectorStore()
        self.llm = OllamaClient()

    def save_uploads(self, files: Iterable) -> list[Path]:
        """Save Streamlit/FastAPI uploaded files to the upload directory."""
        saved_paths: list[Path] = []
        for file in files:
            name = getattr(file, "name", None) or getattr(file, "filename", None)
            if not name:
                continue
            ext = Path(name).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                raise ValueError(f"Unsupported file: {name}. Supported files: {sorted(SUPPORTED_EXTENSIONS)}")
            target = settings.upload_dir / Path(name).name
            if hasattr(file, "getbuffer"):
                target.write_bytes(file.getbuffer())
            elif hasattr(file, "file"):
                with target.open("wb") as f:
                    shutil.copyfileobj(file.file, f)
            else:
                raise ValueError(f"Cannot read uploaded file: {name}")
            saved_paths.append(target)
        return saved_paths

    def ingest_paths(self, paths: Iterable[str | Path]) -> dict:
        paths = [Path(p) for p in paths]
        documents = load_many(paths)
        chunks = chunk_documents(documents, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
        added = self.store.add_chunks(chunks)
        return {
            "files": [p.name for p in paths],
            "documents_loaded": len(documents),
            "chunks_added": added,
            "total_chunks_in_store": self.store.count(),
        }

    def ingest_uploaded_files(self, files: Iterable) -> dict:
        paths = self.save_uploads(files)
        return self.ingest_paths(paths)

    def ask(self, question: str, top_k: int | None = None) -> dict:
        hits = self.store.search(question, top_k=top_k or settings.top_k)
        prompt = build_rag_prompt(question, hits)
        response = self.llm.generate(prompt)
        return {
            "question": question,
            "answer": response.answer,
            "model": response.model,
            "ok": response.ok,
            "sources": hits,
        }

    def match_cv_to_job(self, job_description: str, top_k: int | None = None) -> dict:
        query = "candidate CV profile research experience skills publications projects"
        hits = self.store.search(query, top_k=top_k or settings.top_k)
        prompt = build_cv_match_prompt(job_description, hits)
        response = self.llm.generate(prompt)
        return {
            "job_description": job_description,
            "answer": response.answer,
            "model": response.model,
            "ok": response.ok,
            "sources": hits,
        }

    def reset(self) -> None:
        self.store.reset()
