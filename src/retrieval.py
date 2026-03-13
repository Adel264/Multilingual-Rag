from __future__ import annotations

from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import settings
from .utils import read_jsonl, normalize_text


class Retriever:
    def __init__(self) -> None:
        self.model = SentenceTransformer(settings.embed_model)
        self.index = faiss.read_index(settings.index_path)
        self.metadata = read_jsonl(settings.metadata_path)

    def _encode_query(self, query: str) -> np.ndarray:
        vector = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vector.astype("float32")

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict]:
        top_k = top_k or settings.top_k
        query_vec = self._encode_query(query)
        scores, indices = self.index.search(query_vec, top_k)

        q_norm = normalize_text(query).lower()
        q_tokens = set(q_norm.split())

        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            item = dict(self.metadata[idx])
            item["score"] = float(score)

            if filters:
                valid = True
                for k, v in filters.items():
                    if str(item.get(k, "")).lower() != str(v).lower():
                        valid = False
                        break
                if not valid:
                    continue

            item_q = normalize_text(str(item.get("question", ""))).lower()
            item_q_tokens = set(item_q.split())

            exact_bonus = 0.25 if item_q == q_norm else 0.0
            overlap_bonus = 0.01 * len(q_tokens.intersection(item_q_tokens))

            item["rerank_score"] = item["score"] + exact_bonus + overlap_bonus
            results.append(item)

        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        return results