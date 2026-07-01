import json

import pytest

from obsidian_rag.server import embed_and_store, index_directory, resolve_text


def test_resolve_text_returns_inline_text():
    assert resolve_text("ref", "inline body") == "inline body"


def test_resolve_text_reads_file(tmp_path):
    md = tmp_path / "note.md"
    md.write_text("file body", encoding="utf-8")
    assert resolve_text(str(md), None) == "file body"


def test_resolve_text_missing_file_raises():
    with pytest.raises(ValueError):
        resolve_text("/no/such/file.md", None)


def test_embed_and_store_returns_json_and_writes(isolated_store):
    result = json.loads(embed_and_store("ref-1", "some text"))

    assert result["reference"] == "ref-1"
    assert result["dimensions"] == 256
    assert result["model"] == "feature-hash-256"
    assert "id" in result
    assert result["store_path"].endswith("embeddings.jsonl")
    assert len(isolated_store.read_text(encoding="utf-8").splitlines()) == 1


def test_index_directory_tool_summary(tmp_path, isolated_store):
    (tmp_path / "a.md").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.md").write_text("beta", encoding="utf-8")

    result = json.loads(index_directory(str(tmp_path)))
    assert result["indexed"] == 2
    assert result["skipped"] == 0
    assert result["total"] == 2
    assert result["model"] == "feature-hash-256"


def test_index_directory_tool_rejects_non_directory(tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(ValueError):
        index_directory(str(missing))
