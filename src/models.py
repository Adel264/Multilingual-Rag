from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, Integer, String, Text, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship

from .database import Base


class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), index=True, nullable=True)
    question = Column(Text, nullable=False)
    top_k = Column(Integer, default=5)
    answer = Column(Text, nullable=True)
    latency_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    responses = relationship("RetrievedContext", back_populates="query")


class RetrievedContext(Base):
    __tablename__ = "retrieved_contexts"

    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(Integer, ForeignKey("query_logs.id"), nullable=False)
    chunk_id = Column(String(200), nullable=True)
    source_title = Column(String(255), nullable=True)
    score = Column(Float, nullable=True)
    rerank_score = Column(Float, nullable=True)
    chunk_text = Column(Text, nullable=True)

    query = relationship("QueryLog", back_populates="responses")