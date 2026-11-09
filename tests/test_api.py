import json

from fastapi.testclient import TestClient

from app.config import Settings
from app.llm import AnswerError, build_messages
from app.main import create_app


def _parse(client, data, name="handbuch.pdf"):
    return client.post("/api/parse", files={"file": (name, data, "application/pdf")})


def _events(response):
    events = []
    for block in response.text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def _chunks_with_embeddings(client, sample_pdf, embedder):
    chunks = _parse(client, sample_pdf).json()["chunks"]
    for c, emb in zip(chunks, embedder.embed([c["text"] for c in chunks])):
        c["embedding"] = emb
    return chunks


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok", "password_required": False}


def test_parse_returns_chunks_with_pages(client, sample_pdf):
    res = _parse(client, sample_pdf)
    assert res.status_code == 200
    body = res.json()
    assert body["filename"] == "handbuch.pdf"
    assert body["pages"] == 6
    assert len(body["chunks"]) >= 6
    assert {c["page"] for c in body["chunks"]} == {1, 2, 3, 4, 5, 6}
    assert all(c["doc_id"] == body["id"] for c in body["chunks"])


def test_same_pdf_gets_same_id(client, sample_pdf):
    assert (
        _parse(client, sample_pdf).json()["id"] == _parse(client, sample_pdf, "x.pdf").json()["id"]
    )


def test_parse_rejects_wrong_type_and_broken_pdf(client):
    assert _parse(client, b"hallo", "notiz.txt").status_code == 400
    res = _parse(client, b"%PDF-kaputt")
    assert res.status_code == 422
    assert "PDF" in res.json()["detail"]


def test_chat_streams_sources_then_answer(client, sample_pdf, embedder, generator):
    question = "Wie viele Urlaubstage habe ich?"
    res = client.post(
        "/api/chat",
        json={
            "question": question,
            "query_embedding": embedder.embed([question])[0],
            "chunks": _chunks_with_embeddings(client, sample_pdf, embedder),
        },
    )
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")

    events = _events(res)
    assert [e for e, _ in events] == ["sources", "delta", "delta", "done"]
    sources = events[0][1]["sources"]
    assert sources[0]["n"] == 1
    assert sources[0]["page"] == 3
    assert "".join(d["text"] for e, d in events if e == "delta").endswith("Antwort [1]")
    assert generator.calls[0]["question"] == question


def test_chat_works_without_embeddings(client, sample_pdf):
    chunks = _parse(client, sample_pdf).json()["chunks"]
    res = client.post("/api/chat", json={"question": "Urlaubstage", "chunks": chunks})
    assert _events(res)[0][1]["sources"][0]["page"] == 3


def test_chat_without_documents(client, generator):
    events = _events(client.post("/api/chat", json={"question": "Hallo?"}))
    assert events[0] == ("sources", {"sources": []})
    assert generator.calls[0]["sources"] == []


def test_chat_reports_generator_errors_as_event(client, generator):
    def broken(*_args):
        raise AnswerError("API-Key ungültig")
        yield

    generator.stream = broken
    events = _events(client.post("/api/chat", json={"question": "Hallo?"}))
    assert events[-2] == ("error", {"message": "API-Key ungültig"})
    assert events[-1][0] == "done"


def test_chat_validates_question(client):
    assert client.post("/api/chat", json={"question": ""}).status_code == 422


def test_demo_password(generator):
    client = TestClient(create_app(Settings(_env_file=None, demo_password="geheim"), generator))
    assert client.get("/api/health").json()["password_required"] is True

    assert client.post("/api/chat", json={"question": "Hi"}).status_code == 401
    wrong = {"X-Demo-Password": "falsch"}
    assert client.post("/api/chat", json={"question": "Hi"}, headers=wrong).status_code == 401
    right = {"X-Demo-Password": "geheim"}
    assert client.post("/api/chat", json={"question": "Hi"}, headers=right).status_code == 200


def test_index_page_is_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "pdf-chat" in res.text


def test_build_messages_trims_history_and_starts_with_user():
    history = [{"role": "assistant", "content": "Hallo!"}] + [
        {"role": r, "content": f"m{i}"} for i, r in enumerate(["user", "assistant"] * 5)
    ]
    messages = build_messages("Neue Frage", [], history)
    assert messages[0]["role"] == "user"
    assert len(messages) <= 7
    assert messages[-1]["content"].endswith("Question: Neue Frage")
    assert "no matching excerpts" in messages[-1]["content"]
