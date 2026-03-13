from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import normalize_text


QUESTION_COLUMNS = ["question", "query", "prompt"]
SHORT_ANSWER_COLUMNS = ["short_answer", "answer", "short_answers"]
LONG_ANSWER_COLUMNS = ["long_answers", "long_answer", "context", "passage", "document_text"]
TITLE_COLUMNS = ["source_title", "title", "document_title"]
DOMAIN_COLUMNS = ["domain", "category", "topic"]
LANG_COLUMNS = ["language", "lang"]


@dataclass
class ChunkConfig:
    chunk_size: int = 450
    overlap: int = 80


def _pick_value(row: pd.Series, candidates: list[str], default: str = "") -> str:
    for col in candidates:
        if col in row and pd.notna(row[col]):
            value = str(row[col]).strip()
            if value:
                return value
    return default


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".jsonl", ".json"}:
        return pd.read_json(path, lines=path.suffix.lower() == ".jsonl")

    raise ValueError("Supported formats: .csv, .json, .jsonl")


def classify_difficulty(answer_text: str) -> str:
    length = len(answer_text.split())
    if length <= 12:
        return "easy"
    if length <= 40:
        return "medium"
    return "hard"


def answer_type(short_answer: str, long_answer: str) -> str:
    if short_answer and long_answer:
        return "short_and_long"
    if long_answer:
        return "long_only"
    if short_answer:
        return "short_only"
    return "unknown"


def split_text(text: str, cfg: ChunkConfig) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []

    words = text.split()
    if len(words) <= cfg.chunk_size:
        return [text]

    chunks: list[str] = []
    step = max(cfg.chunk_size - cfg.overlap, 1)
    for start in range(0, len(words), step):
        end = start + cfg.chunk_size
        chunk_words = words[start:end]
        if not chunk_words:
            continue
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
    return chunks


def build_processed_chunks(df: pd.DataFrame, cfg: ChunkConfig) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for row_id, row in df.iterrows():
        question = normalize_text(_pick_value(row, QUESTION_COLUMNS))
        short_answer = normalize_text(_pick_value(row, SHORT_ANSWER_COLUMNS))
        long_answer = normalize_text(_pick_value(row, LONG_ANSWER_COLUMNS))
        title = normalize_text(_pick_value(row, TITLE_COLUMNS, "unknown_source"))
        domain = normalize_text(_pick_value(row, DOMAIN_COLUMNS, "general"))
        language = normalize_text(_pick_value(row, LANG_COLUMNS, "unknown"))

        base_text = long_answer or short_answer
        if not question or not base_text:
            continue

        chunks = split_text(base_text, cfg)
        a_type = answer_type(short_answer, long_answer)
        difficulty = classify_difficulty(base_text)

        for chunk_id, chunk in enumerate(chunks):
            search_text = f"Question: {question}\nShort answer: {short_answer}\nContext: {chunk}"

            records.append(
                {
                    "doc_id": f"doc_{row_id}",
                    "chunk_id": f"doc_{row_id}_chunk_{chunk_id}",
                    "question": question,
                    "short_answer": short_answer,
                    "long_answer": long_answer,
                    "chunk_text": chunk,
                    "search_text": search_text,
                    "source_title": title,
                    "domain": domain,
                    "language": language,
                    "difficulty": difficulty,
                    "answer_type": a_type,
                    "chunk_index": chunk_id,
                }
            )

    return pd.DataFrame(records)


def preprocess_dataset(
    input_path: str | Path,
    output_path: str | Path,
    chunk_size: int = 450,
    overlap: int = 80,
) -> pd.DataFrame:
    df = load_dataset(input_path)
    processed = build_processed_chunks(df, ChunkConfig(chunk_size=chunk_size, overlap=overlap))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_path, index=False)
    return processed