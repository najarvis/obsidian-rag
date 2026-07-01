import hashlib
import json

from obsidian_rag.storage import (
    append_record,
    build_record,
    get_store_path,
    load_existing_hashes,
)


def test_get_store_path_honors_env(monkeypatch, tmp_path):
    target = tmp_path / "custom.jsonl"
    monkeypatch.setenv("EMBEDDINGS_STORE_PATH", str(target))
    assert get_store_path() == target


def test_build_record_has_expected_fields():
    content = "some note content"
    record = build_record("ref-1", content)

    assert set(record) == {
        "id",
        "reference",
        "text_hash",
        "text_preview",
        "embedding",
        "model",
        "created_at",
    }
    assert record["reference"] == "ref-1"
    assert record["text_hash"] == hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert record["model"] == "feature-hash-256"
    assert len(record["embedding"]) == 256


def test_build_record_truncates_preview():
    content = "x" * 500
    record = build_record("ref", content)
    assert record["text_preview"] == "x" * 200


def test_append_record_writes_roundtrippable_line(isolated_store):
    record = build_record("ref", "hello")
    append_record(record)

    lines = isolated_store.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == record


def test_append_record_creates_parent_dir(monkeypatch, tmp_path):
    nested = tmp_path / "deep" / "nested" / "store.jsonl"
    monkeypatch.setenv("EMBEDDINGS_STORE_PATH", str(nested))
    append_record(build_record("ref", "content"))
    assert nested.is_file()


def test_load_existing_hashes_missing_file():
    assert load_existing_hashes() == set()


def test_load_existing_hashes_reads_stored_hashes(isolated_store):
    append_record(build_record("a", "alpha"))
    append_record(build_record("b", "beta"))

    expected = {
        hashlib.sha256(b"alpha").hexdigest(),
        hashlib.sha256(b"beta").hexdigest(),
    }
    assert load_existing_hashes() == expected


def test_load_existing_hashes_skips_blank_and_malformed(isolated_store):
    good_hash = hashlib.sha256(b"good").hexdigest()
    isolated_store.write_text(
        "\n"
        + json.dumps({"text_hash": good_hash})
        + "\n"
        + "not json at all\n"
        + json.dumps({"no_hash_here": True})
        + "\n",
        encoding="utf-8",
    )
    assert load_existing_hashes() == {good_hash}
