import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from step1_ingest import process_book
from step2_embed_and_store import build_index
from step3_retriever import Retriever
from step4_generator import Generator


class RAGSystem:
    def __init__(self, data_dir: str = "data", model: str = "llama3.2"):
        self.data_dir = data_dir
        self.chroma_dir = os.path.join(data_dir, "chroma_db")
        self._retriever = None
        self._generator = Generator()

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = Retriever(chroma_dir=self.chroma_dir)
        return self._retriever

    def ingest_book(self, pdf_path: str) -> None:
        print(f"\nIngesting: {os.path.basename(pdf_path)}")
        chunks_file = process_book(pdf_path, output_dir=self.data_dir)
        build_index(chunks_file, chroma_dir=self.chroma_dir)
        self._retriever = None
        print("Book ready!")

    def ask(self, query: str, top_k: int = 5) -> dict:
        chunks = self.retriever.retrieve(query, top_k=top_k)
        return self._generator.generate(query, chunks)

    def interactive(self) -> None:
        print("\n" + "="*60)
        print("AI Books RAG — Interactive Mode")
        print("Type 'quit' to exit")
        print("="*60)
        while True:
            query = input("\nYour question: ").strip()
            if query.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if not query:
                continue
            print("Searching...")
            result = self.ask(query)
            print(f"\nAnswer:\n{result['answer']}")
            sources = [f"p.{s['page']}" for s in result['sources']]
            print(f"\nSources: {', '.join(sources)}")

from typing import Dict
if __name__ == "__main__":
    import argparse


    parser = argparse.ArgumentParser(description="AI Books RAG")
    parser.add_argument("--ingest", type=str, help="Path to PDF")
    parser.add_argument("--ask", type=str, help="Question to ask")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--model", default="llama3.2")
    args = parser.parse_args()

    rag = RAGSystem(model=args.model)

    if args.ingest:
        rag.ingest_book(args.ingest)

    if args.ask:
        result = rag.ask(args.ask)
        print(f"\nAnswer:\n{result['answer']}")
        for s in result["sources"]:
            print(f"  Page {s['page']} — score: {s['score']:.3f}")

    if args.interactive:
        rag.interactive()

    if not any([args.ingest, args.ask, args.interactive]):
        print("\nUsage:")
        print("  python src/main.py --ingest data/books/book.pdf")
        print("  python src/main.py --ask 'How does B-Tree work?'")
        print("  python src/main.py --interactive")