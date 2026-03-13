from __future__ import annotations

from typing import Iterable

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


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


def evaluate_generation(generation_pairs: list[dict]) -> dict:
    """
    generation_pairs format:
    [
        {"reference": "...", "prediction": "..."},
        ...
    ]
    """
    if not generation_pairs:
        return {
            "bleu": 0.0,
            "rouge1_f1": 0.0,
            "rougeL_f1": 0.0,
            "num_generation_samples": 0,
        }

    smoothie = SmoothingFunction().method1
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    bleu_scores = []
    rouge1_scores = []
    rougeL_scores = []

    for pair in generation_pairs:
        reference = str(pair.get("reference", "")).strip()
        prediction = str(pair.get("prediction", "")).strip()

        if not reference or not prediction:
            continue

        bleu = sentence_bleu(
            [reference.split()],
            prediction.split(),
            smoothing_function=smoothie,
        )
        rouge_scores = scorer.score(reference, prediction)

        bleu_scores.append(bleu)
        rouge1_scores.append(rouge_scores["rouge1"].fmeasure)
        rougeL_scores.append(rouge_scores["rougeL"].fmeasure)

    n = len(bleu_scores)
    if n == 0:
        return {
            "bleu": 0.0,
            "rouge1_f1": 0.0,
            "rougeL_f1": 0.0,
            "num_generation_samples": 0,
        }

    return {
        "bleu": sum(bleu_scores) / n,
        "rouge1_f1": sum(rouge1_scores) / n,
        "rougeL_f1": sum(rougeL_scores) / n,
        "num_generation_samples": n,
    }