from unittest.mock import MagicMock

from scripts import viewer


def _fake_collection(name, rows):
    collection = MagicMock()
    collection.name = name
    collection.get.return_value = rows
    return collection


def test_view_collections_renders_each_collection(monkeypatch):
    rows = {"ids": ["1"], "embeddings": [[0.1]], "metadatas": [{}], "documents": ["doc text"]}
    client = MagicMock()
    client.list_collections.return_value = [_fake_collection("iollamas", rows)]

    monkeypatch.setattr(viewer.chromadb, "PersistentClient", MagicMock(return_value=client))
    monkeypatch.setattr(viewer.st, "markdown", MagicMock())
    monkeypatch.setattr(viewer.st, "header", MagicMock())
    monkeypatch.setattr(viewer.st, "dataframe", MagicMock())

    viewer.view_collections("./some/db/path")

    viewer.chromadb.PersistentClient.assert_called_once_with(path="./some/db/path")
    viewer.st.dataframe.assert_called_once()


def test_view_collections_with_no_collections_does_not_render_any_dataframe(monkeypatch):
    client = MagicMock()
    client.list_collections.return_value = []

    monkeypatch.setattr(viewer.chromadb, "PersistentClient", MagicMock(return_value=client))
    monkeypatch.setattr(viewer.st, "markdown", MagicMock())
    monkeypatch.setattr(viewer.st, "header", MagicMock())
    monkeypatch.setattr(viewer.st, "dataframe", MagicMock())

    viewer.view_collections("./empty/db")

    viewer.st.dataframe.assert_not_called()


def test_errors_are_not_silently_swallowed(monkeypatch):
    """Regression test: the original script had a bare `except: pass` around
    the whole CLI entrypoint that hid every error. view_collections() itself
    must propagate failures instead of swallowing them."""
    monkeypatch.setattr(
        viewer.chromadb, "PersistentClient", MagicMock(side_effect=RuntimeError("db unreachable"))
    )

    try:
        viewer.view_collections("./bad/path")
    except RuntimeError as exc:
        assert "db unreachable" in str(exc)
    else:
        raise AssertionError("expected view_collections to propagate the error")
