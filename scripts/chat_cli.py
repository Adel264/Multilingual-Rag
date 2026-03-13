from __future__ import annotations

from src.rag_pipeline import RAGPipeline


def main() -> None:
    rag = RAGPipeline()
    history: list[dict] = []
    print("Multilingual RAG CLI. Type 'exit' to quit.")
    while True:
        query = input("\nQuestion: ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        result = rag.ask(query=query, top_k=10, history=history)
        print("\nAnswer:\n", result["answer"])
        print("\nTop contexts:")
        for i, item in enumerate(result["retrieved"], start=1):
            print(f"{i}. score={item['score']:.3f} | source={item.get('source_title')} | chunk={item['chunk_text'][:120]}...")
        history.append({"user": query, "assistant": result["answer"]})


if __name__ == "__main__":
    main()
