import pytest

from obsidian_rag.embeddings import embed_text, get_dim, get_model_name, tokenize


def test_embed_text_default_dimension():
    assert len(embed_text("hello world")) == 256


def test_embed_text_is_l2_normalized():
    vec = embed_text("the quick brown fox jumps over the lazy dog")
    norm = sum(value * value for value in vec) ** 0.5
    assert norm == pytest.approx(1.0)


def test_embed_text_is_deterministic():
    assert embed_text("repeatable input") == embed_text("repeatable input")


def test_embed_empty_text_is_zero_vector():
    vec = embed_text("")
    assert len(vec) == 256
    assert all(value == 0.0 for value in vec)


def test_embed_whitespace_only_is_zero_vector():
    assert all(value == 0.0 for value in embed_text("   \n\t  "))


def test_embedding_dim_env_override(monkeypatch):
    monkeypatch.setenv("EMBEDDING_DIM", "64")
    assert get_dim() == 64
    assert len(embed_text("hello world")) == 64
    assert get_model_name() == "feature-hash-64"


def test_get_model_name_default():
    assert get_model_name() == "feature-hash-256"


def test_tokenize_lowercases_and_splits():
    assert tokenize("Hello, WORLD!") == ["hello", "world"]


def test_tokenize_handles_underscores_and_digits():
    assert tokenize("foo_bar baz123") == ["foo_bar", "baz123"]
