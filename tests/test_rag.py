from unittest.mock import MagicMock

import pytest


def _patch_build_index(engine, response_text: str):
    """Stub out the chromadb/llama_index plumbing at RagEngine's own seam.

    Avoids mocking deep into llama_index/chromadb internals (brittle across
    SDK versions); instead fakes the one method RagEngine itself defines.
    """
    fake_query_engine = MagicMock()
    fake_query_engine.query.return_value.response = response_text

    fake_index = MagicMock()
    fake_index.as_query_engine.return_value = fake_query_engine

    engine._build_index = MagicMock(return_value=fake_index)
    return fake_query_engine


def test_chat_before_initialize_raises(rag_engine):
    with pytest.raises(RuntimeError):
        rag_engine.chat("hello", user_id="u1")


def test_initialize_builds_index_once_and_chat_returns_response(rag_engine):
    _patch_build_index(rag_engine, "As per my understanding, ...")

    rag_engine.initialize()
    answer = rag_engine.chat("what is open5gs?", user_id="u1")

    assert answer == "As per my understanding, ..."
    rag_engine._build_index.assert_called_once()


def test_initialize_is_idempotent(rag_engine):
    _patch_build_index(rag_engine, "answer")

    rag_engine.initialize()
    rag_engine.initialize()
    rag_engine.initialize()

    rag_engine._build_index.assert_called_once()


def test_chat_reuses_the_same_query_engine_across_calls(rag_engine):
    """Regression test: the original implementation rebuilt the whole index
    on every single chat() call. It must now be built once and reused."""
    _patch_build_index(rag_engine, "answer")
    rag_engine.initialize()

    rag_engine.chat("question 1", user_id="u1")
    rag_engine.chat("question 2", user_id="u1")
    rag_engine.chat("question 3", user_id="u2")

    rag_engine._build_index.assert_called_once()
