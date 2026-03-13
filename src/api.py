from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .evaluation import evaluate_retrieval
from .rag_pipeline import RAGPipeline

app = FastAPI(title="Multilingual RAG Task API", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=2)
    top_k: int = Field(default=5, ge=1, le=20)
    filters: dict | None = None
    history: list[dict] | None = None


class EvalRequest(BaseModel):
    samples: list[dict]
    k: int = Field(default=5, ge=1, le=20)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask-question")
def ask_question(payload: AskRequest) -> dict:
    try:
        rag = RAGPipeline()
        return rag.ask(
            query=payload.question,
            top_k=payload.top_k,
            filters=payload.filters,
            history=payload.history,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=f"Index not found. Build the index first. {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/evaluate")
def evaluate(payload: EvalRequest) -> dict:
    try:
        return evaluate_retrieval(payload.samples, k=payload.k)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
