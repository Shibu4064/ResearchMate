from __future__ import annotations

import json
from dataclasses import dataclass

import requests

from .config import settings


@dataclass
class LLMResponse:
    answer: str
    model: str
    ok: bool


class OllamaClient:
    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model

    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def generate(self, prompt: str) -> LLMResponse:
        if not self.is_available():
            return LLMResponse(
                ok=False,
                model=self.model,
                answer=(
                    "Ollama is not reachable. Start it with `ollama serve` and pull a model, "
                    "for example `ollama pull llama3.2:3b`."
                ),
            )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": settings.temperature,
                "num_predict": settings.max_tokens,
            },
        }
        try:
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=180)
            response.raise_for_status()
            data = response.json()
            return LLMResponse(answer=data.get("response", "").strip(), model=self.model, ok=True)
        except requests.RequestException as exc:
            return LLMResponse(
                ok=False,
                model=self.model,
                answer=f"LLM generation failed: {exc}",
            )


def build_rag_prompt(question: str, context_blocks: list[dict]) -> str:
    context_parts = []
    total_chars = 0
    for i, block in enumerate(context_blocks, start=1):
        source = block.get("source", "unknown")
        page = block.get("page")
        page_label = f", page {page}" if page else ""
        header = f"[Source {i}: {source}{page_label}]"
        body = block.get("text", "")
        piece = f"{header}\n{body}"
        if total_chars + len(piece) > settings.max_context_chars:
            break
        total_chars += len(piece)
        context_parts.append(piece)

    context = "\n\n".join(context_parts)
    return f"""
You are ResearchMate-RAG, a careful source-grounded research assistant.
Answer the user's question using ONLY the context below.

Rules:
- If the answer is not supported by the context, say that the uploaded documents do not provide enough evidence.
- Cite sources inline using [Source 1], [Source 2], etc.
- Be concise, specific, and academic.
- Do not invent statistics, names, results, or claims.

Context:
{context}

Question:
{question}

Answer:
""".strip()


def build_cv_match_prompt(job_description: str, context_blocks: list[dict]) -> str:
    question = f"""
Compare the candidate's uploaded CV/profile/research documents with the following job description. Produce:
1. Overall fit summary
2. Strong matches
3. Missing or weak areas
4. Suggested CV keywords
5. A short tailored profile summary

Job description:
{job_description}
""".strip()
    return build_rag_prompt(question, context_blocks)
