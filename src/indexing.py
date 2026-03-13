from __future__ import annotations

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .config import settings
from .utils import ensure_parent, write_jsonl


class VectorIndexer:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embed_model
        self.model = SentenceTransformer(self.model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return vectors.astype("float32")

    def build_index(self, processed_df: pd.DataFrame) -> tuple[faiss.Index, list[dict]]:
        texts = processed_df["search_text"].fillna("").tolist()
        embeddings = self.encode(texts)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        metadata = processed_df.to_dict(orient="records")
        return index, metadata

    def save(self, index: faiss.Index, metadata: list[dict]) -> None:
        ensure_parent(settings.index_path)
        faiss.write_index(index, settings.index_path)
        write_jsonl(metadata, settings.metadata_path)
