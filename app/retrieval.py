import re

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
    def __init__(self, store):
        self.store = store
        self.bm25 = None
        self.bm25_chunks = None

    def bm25_search(self, query, k):
        chunks = self.store.all_chunks()
        if not chunks:
            return []
        if self.bm25 is None or self.bm25_chunks is not chunks:
            self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
            self.bm25_chunks = chunks
        scores = self.bm25.get_scores(tokenize(query))
        top = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:k]
        return [ScoredChunk(chunks[i], float(scores[i])) for i in top if scores[i] > 0]

    def search(self, query, k=6, mode="hybrid"):
        if mode == "bm25":
            return self.bm25_search(query, k)
        if mode == "vector":
            return self.store.search(query, k)

        n = max(k * 3, 20)
        vector_hits = [s.chunk for s in self.store.search(query, n)]
        bm25_hits = [s.chunk for s in self.bm25_search(query, n)]
        return reciprocal_rank_fusion([vector_hits, bm25_hits])[:k]
