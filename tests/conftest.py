from unittest.mock import MagicMock

import pytest

from app import create_app


@pytest.fixture
def fake_llm():
    return MagicMock(name="llm")


@pytest.fixture
def fake_embed_model():
    return MagicMock(name="embed_model")


@pytest.fixture
def rag_engine(tmp_path, fake_llm, fake_embed_model):
    """A RagEngine with fake llm/embed_model so construction never calls out
    to Ollama or downloads a HuggingFace model.

    Imported lazily so test files that don't use this fixture (e.g.
    test_routes.py) never need llama_index/chromadb/torch installed.
    """
    from app.rag import RagEngine

    return RagEngine(
        docs_dir=tmp_path / "docs",
        index_persist_directory=tmp_path / "chromadb",
        llm=fake_llm,
        embed_model=fake_embed_model,
    )


class FakeRagEngine:
    """Minimal stand-in for RagEngine used in route tests - no llama_index/chromadb involved."""

    def __init__(self, answer: str = "a canned answer"):
        self.answer = answer
        self.calls = []

    def chat(self, question: str, user_id: str) -> str:
        self.calls.append((question, user_id))
        return self.answer


@pytest.fixture
def fake_rag_engine():
    return FakeRagEngine()


@pytest.fixture
def client(fake_rag_engine):
    app = create_app(rag_engine=fake_rag_engine)
    app.config["TESTING"] = True
    return app.test_client()
