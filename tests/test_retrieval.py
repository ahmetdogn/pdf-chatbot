from app.chunking import chunk_pages
from app.models import Chunk, Page
from app.retrieval import HybridRetriever, reciprocal_rank_fusion, tokenize


def _chunk(i):
    return Chunk(id=i, doc_id="d", filename="a.pdf", page=1, text=i)


def test_rrf_rewards_items_ranked_high_in_both_lists():
    a, b, c = _chunk("a"), _chunk("b"), _chunk("c")
    fused = reciprocal_rank_fusion([[a, b, c], [b, c, a]])
    assert [s.chunk.id for s in fused] == ["b", "a", "c"]


def test_tokenize_drops_stopwords_and_lowercases():
    assert tokenize("Wie viele Urlaubstage hat die Firma?") == ["viele", "urlaubstage", "firma"]


def test_hybrid_search_finds_right_page(embedder):
    pages = [
        Page(1, "Die Kantine öffnet um 11 Uhr und bietet vegetarische Gerichte."),
        Page(2, "Alle Vollzeitkräfte erhalten 30 Urlaubstage pro Jahr."),
        Page(3, "Passwörter müssen mindestens 14 Zeichen lang sein."),
    ]
    chunks = chunk_pages(pages, "doc", "handbuch.pdf")
    retriever = HybridRetriever(chunks, embedder.embed([c.text for c in chunks]))
    query = "Wie viele Urlaubstage gibt es?"
    q_emb = embedder.embed([query])[0]

    for mode in ["bm25", "vector", "hybrid"]:
        top = retriever.search(query, k=1, mode=mode, query_embedding=q_emb)
        assert top[0].chunk.page == 2, mode


def test_bm25_finds_word_that_is_in_half_the_chunks():
    # mit der normalen okapi idf wäre der score hier 0
    chunks = chunk_pages(
        [Page(1, "Erstes Dokument über Katzen."), Page(2, "Zweites Dokument über Hunde.")],
        "d",
        "a.pdf",
    )
    top = HybridRetriever(chunks).search("Hunde", mode="bm25")
    assert top[0].chunk.page == 2


def test_without_embeddings_hybrid_falls_back_to_bm25():
    chunks = chunk_pages([Page(1, "Katzen schlafen viel."), Page(2, "Hunde bellen.")], "d", "a.pdf")
    top = HybridRetriever(chunks).search("Hunde", k=1)
    assert top[0].chunk.page == 2


def test_wrong_embedding_size_is_ignored(embedder):
    chunks = chunk_pages([Page(1, "Hunde bellen.")], "d", "a.pdf")
    retriever = HybridRetriever(chunks, embedder.embed([c.text for c in chunks]))
    assert retriever.search("Hunde", mode="vector", query_embedding=[0.1, 0.2]) == []


def test_search_without_chunks_returns_nothing():
    assert HybridRetriever([]).search("irgendwas") == []
