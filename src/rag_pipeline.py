from __future__ import annotations

from .generation import AnswerGenerator
from .retrieval import Retriever


class RAGPipeline:
    def __init__(self) -> None:
        self.retriever = Retriever()
        self.generator = AnswerGenerator()

    def ask(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
        history: list[dict] | None = None,
    ) -> dict:
        results = self.retriever.search(query=query, top_k=top_k, filters=filters)
        answer = self.generator.generate(query=query, contexts=results, history=history)
        return {
            "query": query,
            "answer": answer,
            "retrieved": results,
        }
