import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.embeddings import HashEmbedder
from app.main import create_app
from eval.sample_content import write_pdf


class FakeGenerator:
    def __init__(self):
        self.calls = []

    def stream(self, question, sources, history):
        self.calls.append({"question": question, "sources": sources, "history": history})
        yield f"{len(sources)} Quellen gefunden. "
        yield "Antwort [1]"


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory):
    path = tmp_path_factory.mktemp("pdf") / "sample.pdf"
    write_pdf(path)
    return path.read_bytes()


@pytest.fixture
def generator():
    return FakeGenerator()


@pytest.fixture
def client(tmp_path, generator):
    settings = Settings(data_dir=tmp_path, embedder="hash", _env_file=None)
    app = create_app(settings, embedder=HashEmbedder(), generator=generator)
    return TestClient(app)
