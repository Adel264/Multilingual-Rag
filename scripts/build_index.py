from __future__ import annotations

import argparse
from pathlib import Path

from src.config import settings
from src.data_processing import preprocess_dataset
from src.indexing import VectorIndexer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw dataset (.csv/.json/.jsonl)")
    parser.add_argument("--output", default=settings.processed_data_path)
    parser.add_argument("--chunk-size", type=int, default=450)
    parser.add_argument("--overlap", type=int, default=80)
    args = parser.parse_args()

    processed = preprocess_dataset(
        input_path=args.input,
        output_path=args.output,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    if processed.empty:
        raise ValueError("No valid rows were produced after preprocessing.")

    indexer = VectorIndexer()
    index, metadata = indexer.build_index(processed)
    indexer.save(index, metadata)

    print(f"Processed rows: {len(processed)}")
    print(f"Saved index to: {Path(settings.index_path).resolve()}")
    print(f"Saved metadata to: {Path(settings.metadata_path).resolve()}")


if __name__ == "__main__":
    main()
