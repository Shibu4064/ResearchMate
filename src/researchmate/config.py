from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Central configuration for the ResearchMate-RAG system."""

    app_name: str = "ResearchMate-RAG"
    upload_dir: Path = PROJECT_ROOT / "data" / "uploads"
    chroma_dir: Path = PROJECT_ROOT / "chroma_store"
    collection_name: str = os.getenv("CHROMA_COLLECTION", "researchmate_documents")

    # Embedding model: small, fast, and good enough for local Mac usage.
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # Ollama local LLM settings. Recommended on Apple Silicon: llama3.2:3b, qwen2.5:3b, mistral, llama3.1:8b.
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

    # Retrieval settings.
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "850"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "140"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    max_context_chars: int = int(os.getenv("MAX_CONTEXT_CHARS", "9000"))

    # Generation settings.
    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "900"))


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.chroma_dir.mkdir(parents=True, exist_ok=True)
