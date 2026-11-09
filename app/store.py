from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.models import Chunk, ScoredChunk


def to_chunk(id, text, meta):
    return Chunk(id, meta["doc_id"], meta["filename"], meta["page"], text)


class VectorStore:
    def __init__(self, path: Path, embedder):
        path.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        # name mit dim, damit man beim modellwechsel nicht vektoren mischt
        self.col = client.get_or_create_collection(
            f"chunks_{embedder.dim}", metadata={"hnsw:space": "cosine"}
        )
        self._all = None

    def add(self, chunks: list[Chunk]):
        for i in range(0, len(chunks), 256):
            batch = chunks[i : i + 256]
            self.col.add(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                embeddings=self.embedder.embed([c.text for c in batch]),
                metadatas=[
                    {"doc_id": c.doc_id, "filename": c.filename, "page": c.page} for c in batch
                ],
            )
        self._all = None

    def has_document(self, doc_id):
        return len(self.col.get(where={"doc_id": doc_id}, limit=1)["ids"]) > 0

    def delete_document(self, doc_id):
        if not self.has_document(doc_id):
            return False
        self.col.delete(where={"doc_id": doc_id})
        self._all = None
        return True

    def all_chunks(self) -> list[Chunk]:
        if self._all is None:
            res = self.col.get(include=["documents", "metadatas"])
            self._all = [
                to_chunk(*row) for row in zip(res["ids"], res["documents"], res["metadatas"])
            ]
        return self._all

    def documents(self):
        docs = {}
        for c in self.all_chunks():
            if c.doc_id not in docs:
                docs[c.doc_id] = {"id": c.doc_id, "filename": c.filename}
                docs[c.doc_id]["pages"] = set()
                docs[c.doc_id]["chunks"] = 0
            docs[c.doc_id]["pages"].add(c.page)
            docs[c.doc_id]["chunks"] += 1
        for d in docs.values():
            d["pages"] = len(d["pages"])
        return list(docs.values())

    def search(self, query, k) -> list[ScoredChunk]:
        n = self.col.count()
        if n == 0:
            return []
        res = self.col.query(
            query_embeddings=self.embedder.embed([query]),
            n_results=min(k, n),
            include=["documents", "metadatas", "distances"],
        )
        rows = zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
        return [ScoredChunk(to_chunk(i, d, m), 1 - dist) for i, d, m, dist in rows]
