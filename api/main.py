from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.researchmate.rag_pipeline import RAGPipeline

app = FastAPI(title="ResearchMate-RAG API", version="1.0.0")
pipeline = RAGPipeline()


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class JobMatchRequest(BaseModel):
    job_description: str
    top_k: int = 5


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "chunks_in_store": pipeline.store.count(),
        "ollama_available": pipeline.llm.is_available(),
    }


@app.post("/ingest")
async def ingest(files: Annotated[list[UploadFile], File(...)]) -> dict:
    try:
        return pipeline.ingest_uploaded_files(files)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/ask")
def ask(payload: AskRequest) -> dict:
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    return pipeline.ask(payload.question, top_k=payload.top_k)


@app.post("/match-cv")
def match_cv(payload: JobMatchRequest) -> dict:
    if not payload.job_description.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty.")
    return pipeline.match_cv_to_job(payload.job_description, top_k=payload.top_k)


@app.post("/reset")
def reset() -> dict:
    pipeline.reset()
    return {"status": "reset complete"}
