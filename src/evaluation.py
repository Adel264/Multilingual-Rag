from __future__ import annotations

from typing import Iterable


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = retrieved_ids[:k]
    if not top:
        return 0.0
    hits = sum(1 for x in top if x in relevant_ids)
    return hits / len(top)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top = retrieved_ids[:k]
    hits = sum(1 for x in top if x in relevant_ids)
    return hits / len(relevant_ids)


def evaluate_retrieval(samples: Iterable[dict], k: int = 5) -> dict:
    precisions = []
    recalls = []
    count = 0
    for sample in samples:
        count += 1
        retrieved_ids = sample["retrieved_ids"]
        relevant_ids = set(sample["relevant_ids"])
        precisions.append(precision_at_k(retrieved_ids, relevant_ids, k))
        recalls.append(recall_at_k(retrieved_ids, relevant_ids, k))

    if count == 0:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "num_samples": 0}

    return {
        "precision_at_k": sum(precisions) / count,
        "recall_at_k": sum(recalls) / count,
        "num_samples": count,
    }
