# testet ob die suche die richtige seite findet (ohne claude, kostet also nix)
# python -m eval.run_eval
# python -m eval.run_eval --embedder hash

import argparse
import json
import tempfile
from pathlib import Path

from app.chunking import chunk_pages
from app.config import Settings
from app.embeddings import build_embedder
from app.pdf_loader import extract_pages
from app.retrieval import HybridRetriever
from app.store import VectorStore
from eval.sample_content import FILENAME, write_pdf


def evaluate(retriever, questions, k):
    results = {}
    for mode in ["bm25", "vector", "hybrid"]:
        hits = 0
        rr = 0
        for q in questions:
            pages = [s.chunk.page for s in retriever.search(q["question"], k=k, mode=mode)]
            if q["page"] in pages:
                hits += 1
                rr += 1 / (pages.index(q["page"]) + 1)
        results[mode] = (hits / len(questions), rr / len(questions))
    return results


def main():
    settings = Settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedder", default=settings.embedder)
    parser.add_argument("-k", type=int, default=3)
    args = parser.parse_args()

    questions = json.loads((Path(__file__).parent / "questions.json").read_text("utf-8"))
    embedder = build_embedder(args.embedder, settings.embedding_model)

    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / FILENAME
        write_pdf(pdf)
        store = VectorStore(Path(tmp) / "chroma", embedder)
        # kleinere chunks, sonst gibts pro seite nur einen und das ranking ist egal
        store.add(chunk_pages(extract_pages(pdf.read_bytes()), "sample", FILENAME, 300, 50))
        results = evaluate(HybridRetriever(store), questions, args.k)

    print(f"\n{len(questions)} Fragen, embedder={args.embedder}, k={args.k}\n")
    print(f"| Modus  | Hit@{args.k} | MRR  |")
    print("|--------|-------|------|")
    for mode, (hit, mrr) in results.items():
        print(f"| {mode:<6} | {hit:>4.0%}  | {mrr:.2f} |")


if __name__ == "__main__":
    main()
