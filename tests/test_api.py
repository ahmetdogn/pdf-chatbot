import json

from app.llm import build_messages


def _upload(client, data: bytes, name: str = "handbuch.pdf"):
    return client.post("/api/documents", files={"file": (name, data, "application/pdf")})


def _events(response):
    events = []
    for block in response.text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_index_page_is_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "pdf-chat" in res.text


def test_upload_list_and_delete(client, sample_pdf):
    res = _upload(client, sample_pdf)
    assert res.status_code == 201
    body = res.json()
    assert body["pages"] == 6 and body["chunks"] >= 6

    docs = client.get("/api/documents").json()
    assert docs == [
        {"id": body["id"], "filename": "handbuch.pdf", "pages": 6, "chunks": body["chunks"]}
    ]

    assert client.delete(f"/api/documents/{body['id']}").status_code == 204
    assert client.get("/api/documents").json() == []
    assert client.delete(f"/api/documents/{body['id']}").status_code == 404


def test_duplicate_upload_is_detected(client, sample_pdf):
    first = _upload(client, sample_pdf).json()
    second = _upload(client, sample_pdf, "kopie.pdf").json()
    assert second == {"id": first["id"], "filename": "kopie.pdf", "duplicate": True}
    assert len(client.get("/api/documents").json()) == 1


def test_upload_rejects_wrong_type_and_broken_pdf(client):
    assert _upload(client, b"hallo", "notiz.txt").status_code == 400
    res = _upload(client, b"%PDF-kaputt")
    assert res.status_code == 422
    assert "PDF" in res.json()["detail"]


def test_chat_streams_sources_then_answer(client, sample_pdf, generator):
    _upload(client, sample_pdf)
    res = client.post("/api/chat", json={"question": "Wie viele Urlaubstage habe ich?"})
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")

    events = _events(res)
    assert [e for e, _ in events] == ["sources", "delta", "delta", "done"]
    sources = events[0][1]["sources"]
    assert sources[0]["n"] == 1
    assert sources[0]["page"] == 3
    assert "".join(d["text"] for e, d in events if e == "delta").endswith("Antwort [1]")
    assert generator.calls[0]["question"] == "Wie viele Urlaubstage habe ich?"


def test_chat_reports_generator_errors_as_event(client, generator):
    from app.llm import AnswerError

    def broken(*_args):
        raise AnswerError("Ungültiger API-Key.")
        yield

    generator.stream = broken
    events = _events(client.post("/api/chat", json={"question": "Hallo?"}))
    assert events[-2] == ("error", {"message": "Ungültiger API-Key."})
    assert events[-1][0] == "done"


def test_chat_validates_question(client):
    assert client.post("/api/chat", json={"question": ""}).status_code == 422


def test_build_messages_trims_history_and_starts_with_user():
    history = [{"role": "assistant", "content": "Hallo!"}] + [
        {"role": r, "content": f"m{i}"} for i, r in enumerate(["user", "assistant"] * 5)
    ]
    messages = build_messages("Neue Frage", [], history)
    assert messages[0]["role"] == "user"
    assert len(messages) <= 7
    assert messages[-1]["content"].endswith("Question: Neue Frage")
    assert "no matching excerpts" in messages[-1]["content"]
