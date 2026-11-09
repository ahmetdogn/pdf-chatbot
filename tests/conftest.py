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
def client(generator):
    settings = Settings(_env_file=None)
    return TestClient(create_app(settings, generator=generator))


@pytest.fixture
def embedder():
    return HashEmbedder()
