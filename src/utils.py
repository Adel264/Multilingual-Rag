from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


_whitespace_re = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.replace("\u200f", " ").replace("\u200e", " ")
    text = _whitespace_re.sub(" ", text).strip()
    return text


def write_jsonl(records: Iterable[dict], path: str | Path) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
