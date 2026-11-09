import math
import re

import numpy as np
from rank_bm25 import BM25Okapi

from app.models import ScoredChunk

STOPWORDS = set(
    """
    der die das den dem des ein eine einer eines einem einen und oder aber in im
    an am auf aus bei mit nach von vor zu zum zur für über unter ist sind war waren
    wird werden wie was wer wo wann warum welche welcher welches ich du er sie es wir
    ihr man nicht kein keine auch noch nur so als wenn dass da sich mir mich dir dich
    hat habe haben hast gibt geht kann können muss müssen darf dürfen soll mein meine
    meinen meiner dein deine bin bist sein
    the a an and or of to in on for is are was were be how what who when where why
    which it this that with as by at from
    """.split()
)


def tokenize(text):
    return [t for t in re.findall(r"\w+", text.lower()) if len(t) > 1 and t not in STOPWORDS]


class BM25(BM25Okapi):
    # die normale okapi idf wird 0 oder negativ wenn ein wort in >= der hälfte der chunks
    # vorkommt. bei kleinen pdfs sind dann genau die richtigen treffer weg.
    # deshalb die idf von lucene, die ist immer positiv
    def _calc_idf(self, nd):
        n = self.corpus_size
        self.idf = {}
        for word, freq in nd.items():
            self.idf[word] = math.log(1 + (n - freq + 0.5) / (freq + 0.5))


def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    chunks = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.id] = scores.get(chunk.id, 0) + 1 / (k + rank)
            chunks[chunk.id] = chunk
    best = sorted(scores, key=scores.get, reverse=True)
    return [ScoredChunk(chunks[i], scores[i]) for i in best]


class HybridRetriever:
    # alles im speicher, der server merkt sich nichts zwischen requests (wegen vercel)
    def __init__(self, chunks, embeddings=None):
        self.chunks = chunks
        self.bm25 = BM25([tokenize(c.text) for c in chunks]) if chunks else None
        self.vectors = None
        if embeddings is not None and len(embeddings) == len(chunks) and chunks:
            vectors = np.array(embeddings, dtype=np.float32)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            self.vectors = vectors / np.maximum(norms, 1e-9)

    def bm25_search(self, query, k):
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(-scores)[:k]
        return [ScoredChunk(self.chunks[i], float(scores[i])) for i in top if scores[i] > 0]

    def vector_search(self, query_embedding, k):
        if self.vectors is None or query_embedding is None:
            return []
        q = np.array(query_embedding, dtype=np.float32)
        if q.shape[0] != self.vectors.shape[1]:
            return []
        scores = self.vectors @ (q / max(np.linalg.norm(q), 1e-9))
        top = np.argsort(-scores)[:k]
        return [ScoredChunk(self.chunks[i], float(scores[i])) for i in top]

    def search(self, query, k=6, mode="hybrid", query_embedding=None):
        if mode == "bm25":
            return self.bm25_search(query, k)
        if mode == "vector":
            return self.vector_search(query_embedding, k)

        n = max(k * 3, 20)
        vector_hits = [s.chunk for s in self.vector_search(query_embedding, n)]
        bm25_hits = [s.chunk for s in self.bm25_search(query, n)]
        return reciprocal_rank_fusion([vector_hits, bm25_hits])[:k]
