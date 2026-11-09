from app.chunking import chunk_pages
from app.embeddings import HashEmbedder
from app.models import Chunk, Page
from app.retrieval import HybridRetriever, reciprocal_rank_fusion, tokenize
from app.store import VectorStore


def _chunk(i: str) -> Chunk:
    return Chunk(id=i, doc_id="d", filename="a.pdf", page=1, text=i)


def test_rrf_rewards_items_ranked_high_in_both_lists():
    a, b, c = _chunk("a"), _chunk("b"), _chunk("c")
    fused = reciprocal_rank_fusion([[a, b, c], [b, c, a]])
    assert [s.chunk.id for s in fused] == ["b", "a", "c"]


def test_tokenize_drops_stopwords_and_lowercases():
    assert tokenize("Wie viele Urlaubstage hat die Firma?") == ["viele", "urlaubstage", "firma"]


def test_hybrid_search_finds_right_page(tmp_path):
    pages = [
        Page(1, "Die Kantine öffnet um 11 Uhr und bietet vegetarische Gerichte."),
        Page(2, "Alle Vollzeitkräfte erhalten 30 Urlaubstage pro Jahr."),
        Page(3, "Passwörter müssen mindestens 14 Zeichen lang sein."),
    ]
    store = VectorStore(tmp_path, HashEmbedder())
    store.add(chunk_pages(pages, "doc", "handbuch.pdf"))
    retriever = HybridRetriever(store)

    for mode in ("bm25", "vector", "hybrid"):
        top = retriever.search("Wie viele Urlaubstage gibt es?", k=1, mode=mode)
        assert top[0].chunk.page == 2, mode


def test_bm25_index_refreshes_after_new_upload(tmp_path):
    store = VectorStore(tmp_path, HashEmbedder())
    retriever = HybridRetriever(store)
    store.add(chunk_pages([Page(1, "Erstes Dokument über Katzen.")], "d1", "a.pdf"))
    assert retriever.search("Hunde", mode="bm25") == []

    store.add(chunk_pages([Page(1, "Zweites Dokument über Hunde.")], "d2", "b.pdf"))
    assert retriever.search("Hunde", mode="bm25")[0].chunk.doc_id == "d2"


def test_search_on_empty_store_returns_nothing(tmp_path):
    retriever = HybridRetriever(VectorStore(tmp_path, HashEmbedder()))
    assert retriever.search("irgendwas") == []
