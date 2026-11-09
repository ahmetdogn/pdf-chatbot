import hashlib
import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.chunking import chunk_pages
from app.config import get_settings
from app.embeddings import build_embedder
from app.llm import AnswerError, ClaudeGenerator
from app.pdf_loader import PdfError, extract_pages
from app.retrieval import HybridRetriever
from app.store import VectorStore

log = logging.getLogger("pdf-chat")
STATIC_DIR = Path(__file__).parent.parent / "static"


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[dict] = []


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(settings=None, embedder=None, generator=None):
    settings = settings or get_settings()
    embedder = embedder or build_embedder(settings.embedder, settings.embedding_model)
    store = VectorStore(settings.data_dir / "chroma", embedder)
    retriever = HybridRetriever(store)

    app = FastAPI(title="pdf-chat")

    # erst beim ersten chat erstellen, damit die app auch ohne api key startet
    def get_generator():
        nonlocal generator
        if generator is None:
            generator = ClaudeGenerator(
                settings.claude_model, settings.claude_effort, settings.claude_max_tokens
            )
        return generator

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/documents")
    def list_documents():
        return store.documents()

    @app.post("/api/documents", status_code=201)
    async def upload(file: UploadFile):
        filename = Path(file.filename or "dokument.pdf").name
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(400, "Bitte nur PDFs hochladen")

        data = await file.read()
        if len(data) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"Datei zu groß (max {settings.max_upload_mb} MB)")

        doc_id = hashlib.sha256(data).hexdigest()[:16]
        if store.has_document(doc_id):
            return {"id": doc_id, "filename": filename, "duplicate": True}

        try:
            pages = extract_pages(data)
        except PdfError as e:
            raise HTTPException(422, str(e)) from e

        chunks = chunk_pages(pages, doc_id, filename, settings.chunk_size, settings.chunk_overlap)
        store.add(chunks)
        log.info("%s: %d seiten, %d chunks", filename, len(pages), len(chunks))
        return {"id": doc_id, "filename": filename, "pages": len(pages), "chunks": len(chunks)}

    @app.delete("/api/documents/{doc_id}", status_code=204)
    def delete(doc_id: str):
        if not store.delete_document(doc_id):
            raise HTTPException(404, "Dokument nicht gefunden")

    @app.post("/api/chat")
    def chat(req: ChatRequest):
        sources = retriever.search(req.question, k=settings.top_k)

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

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def app_factory():
    logging.basicConfig(level=logging.INFO)
    return create_app()
