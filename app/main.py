import hashlib
import json
import logging
import os
import secrets
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.chunking import chunk_pages
from app.config import get_settings
from app.llm import AnswerError, ClaudeGenerator
from app.models import Chunk
from app.pdf_loader import PdfError, extract_pages
from app.retrieval import HybridRetriever

log = logging.getLogger("pdf-chat")
PUBLIC_DIR = Path(__file__).parent.parent / "public"


class ChunkIn(BaseModel):
    id: str
    doc_id: str
    filename: str
    page: int
    text: str
    embedding: list[float] | None = None


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    query_embedding: list[float] | None = None
    chunks: list[ChunkIn] = Field(default=[], max_length=3000)
    history: list[dict] = []


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(settings=None, generator=None):
    settings = settings or get_settings()
    app = FastAPI(title="pdf-chat")

    # erst beim ersten chat erstellen, damit die app auch ohne api key startet
    def get_generator():
        nonlocal generator
        if generator is None:
            generator = ClaudeGenerator(
                settings.claude_model, settings.claude_effort, settings.claude_max_tokens
            )
        return generator

    def check_password(password):
        if settings.demo_password and not secrets.compare_digest(
            password or "", settings.demo_password
        ):
            raise HTTPException(401, "Falsches Passwort")

    @app.get("/api/health")
    def health():
        return {"status": "ok", "password_required": bool(settings.demo_password)}

    @app.post("/api/parse")
    async def parse(file: UploadFile):
        filename = Path(file.filename or "dokument.pdf").name
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(400, "Bitte nur PDFs hochladen")

        data = await file.read()
        if len(data) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"Datei zu groß (max {settings.max_upload_mb} MB)")

        try:
            pages = extract_pages(data)
        except PdfError as e:
            raise HTTPException(422, str(e)) from e

        doc_id = hashlib.sha256(data).hexdigest()[:16]
        chunks = chunk_pages(pages, doc_id, filename, settings.chunk_size, settings.chunk_overlap)
        log.info("%s: %d seiten, %d chunks", filename, len(pages), len(chunks))
        return {
            "id": doc_id,
            "filename": filename,
            "pages": len(pages),
            "chunks": [c.__dict__ for c in chunks],
        }

    @app.post("/api/chat")
    def chat(req: ChatRequest, x_demo_password: str | None = Header(default=None)):
        check_password(x_demo_password)

        chunks = [Chunk(c.id, c.doc_id, c.filename, c.page, c.text) for c in req.chunks]
        embeddings = None
        if req.chunks and all(c.embedding for c in req.chunks):
            embeddings = [c.embedding for c in req.chunks]
        retriever = HybridRetriever(chunks, embeddings)
        sources = retriever.search(
            req.question, settings.top_k, query_embedding=req.query_embedding
        )

        def events():
            yield sse(
                "sources",
                {
                    "sources": [
                        {
                            "n": n,
                            "doc_id": s.chunk.doc_id,
                            "filename": s.chunk.filename,
                            "page": s.chunk.page,
                            "text": s.chunk.text,
                        }
                        for n, s in enumerate(sources, start=1)
                    ]
                },
            )
            try:
                for text in get_generator().stream(req.question, sources, req.history):
                    yield sse("delta", {"text": text})
            except AnswerError as e:
                yield sse("error", {"message": str(e)})
            except Exception:
                log.exception("chat failed")
                yield sse("error", {"message": "Irgendwas ist schiefgelaufen"})
            yield sse("done", {})

        return StreamingResponse(events(), media_type="text/event-stream")

    # auf vercel kommt public/ direkt vom cdn, lokal liefert fastapi die dateien aus
    if not os.environ.get("VERCEL"):
        app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="public")

    return app


logging.basicConfig(level=logging.INFO)
app = create_app()
