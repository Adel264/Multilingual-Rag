from __future__ import annotations


import time
from sqlalchemy import func
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .database import Base, SessionLocal, engine
from .evaluation import evaluate_retrieval, evaluate_generation
from .models import QueryLog, RetrievedContext
from .rag_pipeline import RAGPipeline

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Multilingual RAG Task API", version="0.2.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=2)
    top_k: int = Field(default=5, ge=1, le=20)
    filters: dict | None = None
    history: list[dict] | None = None
    session_id: str | None = None


class EvalRequest(BaseModel):
    samples: list[dict] = []
    generation_pairs: list[dict] | None = None
    k: int = Field(default=5, ge=1, le=20)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask-question")
def ask_question(payload: AskRequest) -> dict:
    db = SessionLocal()
    start_time = time.perf_counter()

    try:
        rag = RAGPipeline()
        result = rag.ask(
            query=payload.question,
            top_k=payload.top_k,
            filters=payload.filters,
            history=payload.history,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        query_log = QueryLog(
            session_id=payload.session_id,
            question=payload.question,
            top_k=payload.top_k,
            answer=result["answer"],
            latency_ms=latency_ms,
        )
        db.add(query_log)
        db.flush()

        for item in result["retrieved"]:
            ctx = RetrievedContext(
                query_id=query_log.id,
                chunk_id=item.get("chunk_id"),
                source_title=item.get("source_title"),
                score=item.get("score"),
                rerank_score=item.get("rerank_score"),
                chunk_text=item.get("chunk_text"),
            )
            db.add(ctx)

        db.commit()

        result["session_id"] = payload.session_id
        result["log_id"] = query_log.id
        result["latency_ms"] = round(latency_ms, 2)
        return result

    except FileNotFoundError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Index not found. Build the index first. {exc}")
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        db.close()

@app.get("/analytics-summary")
def analytics_summary() -> dict:
    db = SessionLocal()
    try:
        total_queries = db.query(func.count(QueryLog.id)).scalar() or 0
        avg_latency = db.query(func.avg(QueryLog.latency_ms)).scalar() or 0.0
        unique_sessions = db.query(func.count(func.distinct(QueryLog.session_id))).scalar() or 0

        recent_queries = (
            db.query(QueryLog)
            .order_by(QueryLog.created_at.desc())
            .limit(5)
            .all()
        )

        return {
            "total_queries": int(total_queries),
            "average_latency_ms": round(float(avg_latency), 2),
            "unique_sessions": int(unique_sessions),
            "recent_queries": [
                {
                    "id": q.id,
                    "session_id": q.session_id,
                    "question": q.question,
                    "top_k": q.top_k,
                    "latency_ms": round(float(q.latency_ms or 0.0), 2),
                    "created_at": q.created_at.isoformat() if q.created_at else None,
                }
                for q in recent_queries
            ],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        db.close()

@app.get("/benchmark-summary")
def benchmark_summary() -> dict:
    db = SessionLocal()
    try:
        total_queries = db.query(func.count(QueryLog.id)).scalar() or 0
        avg_latency = db.query(func.avg(QueryLog.latency_ms)).scalar() or 0.0
        min_latency = db.query(func.min(QueryLog.latency_ms)).scalar() or 0.0
        max_latency = db.query(func.max(QueryLog.latency_ms)).scalar() or 0.0

        first_query = db.query(QueryLog).order_by(QueryLog.created_at.asc()).first()
        last_query = db.query(QueryLog).order_by(QueryLog.created_at.desc()).first()

        qpm = None

        if (
            total_queries >= 2
            and first_query
            and last_query
            and first_query.created_at
            and last_query.created_at
        ):
            duration_minutes = (
                last_query.created_at - first_query.created_at
            ).total_seconds() / 60.0

            if duration_minutes > 0:
                qpm = total_queries / duration_minutes

        return {
            "total_queries": int(total_queries),
            "avg_latency_ms": round(float(avg_latency), 2),
            "min_latency_ms": round(float(min_latency), 2),
            "max_latency_ms": round(float(max_latency), 2),
            "queries_per_minute_estimate": round(float(qpm), 2) if qpm is not None else None,
            "benchmark_note": "Throughput estimate requires at least 2 logged queries over non-zero elapsed time."
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        db.close()

@app.post("/evaluate")
def evaluate(payload: EvalRequest) -> dict:
    db = SessionLocal()
    try:
        retrieval_metrics = evaluate_retrieval(payload.samples, k=payload.k)

        generation_metrics = evaluate_generation(payload.generation_pairs or [])

        avg_latency = db.query(func.avg(QueryLog.latency_ms)).scalar() or 0.0
        total_queries = db.query(func.count(QueryLog.id)).scalar() or 0

        return {
            "retrieval": retrieval_metrics,
            "generation": generation_metrics,
            "system_performance": {
                "average_latency_ms": round(float(avg_latency), 2),
                "logged_queries": int(total_queries),
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        db.close()