import hashlib

from obsidian_rag.indexer import index_directory, index_file, iter_markdown_files


def _make_vault(root):
    (root / "a.md").write_text("alpha content", encoding="utf-8")
    sub = root / "sub"
    sub.mkdir()
    (sub / "b.md").write_text("beta content", encoding="utf-8")
    (root / "notes.txt").write_text("not markdown", encoding="utf-8")
    return root


def test_iter_markdown_files_recursive_sorted_ignores_non_md(tmp_path):
    _make_vault(tmp_path)
    files = iter_markdown_files(tmp_path)
    names = [p.name for p in files]
    assert names == ["a.md", "b.md"]
    assert files == sorted(files)


def test_index_file_skips_known_hash(tmp_path, isolated_store):
    md = tmp_path / "note.md"
    md.write_text("known content", encoding="utf-8")
    known = {hashlib.sha256(b"known content").hexdigest()}

    assert index_file(md, known) is None
    assert not isolated_store.exists() or isolated_store.read_text() == ""


def test_index_file_stores_new_content(tmp_path, isolated_store):
    md = tmp_path / "note.md"
    md.write_text("fresh content", encoding="utf-8")
    known: set[str] = set()

    record = index_file(md, known)
    assert record is not None
    assert record["text_hash"] in known
    assert len(isolated_store.read_text(encoding="utf-8").splitlines()) == 1


def test_index_directory_counts(tmp_path, isolated_store):
    _make_vault(tmp_path)
    summary = index_directory(tmp_path)
    assert summary == {"indexed": 2, "skipped": 0, "total": 2}
    assert len(isolated_store.read_text(encoding="utf-8").splitlines()) == 2


def test_index_directory_idempotent_rerun(tmp_path, isolated_store):
    _make_vault(tmp_path)
    index_directory(tmp_path)
    summary = index_directory(tmp_path)
    assert summary == {"indexed": 0, "skipped": 2, "total": 2}
    assert len(isolated_store.read_text(encoding="utf-8").splitlines()) == 2
