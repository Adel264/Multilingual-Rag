from __future__ import annotations

from typing import Any
import requests

from .config import settings


class AnswerGenerator:
    def __init__(self) -> None:
        self.api_key = settings.llm_api_key
        self.api_base = settings.llm_api_base.rstrip("/")
        self.model = settings.llm_model

    def build_prompt(self, query: str, contexts: list[dict], history: list[dict] | None = None) -> str:
        history_text = ""
        if history:
            history_lines = []
            for turn in history[-5:]:
                history_lines.append(f"User: {turn.get('user', '')}")
                history_lines.append(f"Assistant: {turn.get('assistant', '')}")
            history_text = "\n".join(history_lines)

        context_text = "\n\n".join(
            [
                f"[Context #{i+1} | score={c['score']:.3f} | source={c.get('source_title', 'unknown')}]\n{c['chunk_text']}"
                for i, c in enumerate(contexts)
            ]
        )

        return f"""
You are a multilingual RAG assistant.
Answer only from the provided context.
If the context is insufficient, say clearly that the answer could not be confirmed.

Conversation history:
{history_text or 'N/A'}

User question:
{query}

Retrieved context:
{context_text}

Return a concise, accurate answer and cite the source title at the end.
""".strip()

    def _extractive_fallback(self, query: str, contexts: list[dict]) -> str:
        if not contexts:
            return "I could not confirm the answer from the retrieved context."

        best = contexts[0]

        if best["score"] < settings.min_score:
            return "I found low-confidence matches only, so I cannot confirm the answer."

        short_answer = str(best.get("short_answer", "")).strip()
        if short_answer and short_answer.lower() != "nan":
            return f"{short_answer}\n\nSource: {best.get('source_title', 'unknown_source')}"

        return f"{best['chunk_text']}\n\nSource: {best.get('source_title', 'unknown_source')}"

    def generate(self, query: str, contexts: list[dict], history: list[dict] | None = None) -> str:
        if not self.api_key:
            return self._extractive_fallback(query, contexts)

        prompt = self.build_prompt(query, contexts, history)
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a careful retrieval-grounded assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            return self._extractive_fallback(query, contexts)
