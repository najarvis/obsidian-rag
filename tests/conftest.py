import pytest


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    store = tmp_path / "embeddings.jsonl"
    monkeypatch.setenv("EMBEDDINGS_STORE_PATH", str(store))
    monkeypatch.delenv("EMBEDDING_DIM", raising=False)
    return store
