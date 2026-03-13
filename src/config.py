from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    embed_model: str = os.getenv(
        "EMBED_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    index_path: str = os.getenv("INDEX_PATH", "artifacts/faiss.index")
    metadata_path: str = os.getenv("METADATA_PATH", "artifacts/metadata.jsonl")
    processed_data_path: str = os.getenv(
        "PROCESSED_DATA_PATH", "data/processed/processed_chunks.csv"
    )
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_api_base: str = os.getenv("LLM_API_BASE", "https://api.groq.com/openai/v1")
    llm_model: str = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    top_k: int = int(os.getenv("TOP_K", "5"))
    min_score: float = float(os.getenv("MIN_SCORE", "0.35"))


settings = Settings()
