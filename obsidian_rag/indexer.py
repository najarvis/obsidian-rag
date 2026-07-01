import sys
import hashlib
from pathlib import Path

from obsidian_rag.storage import append_record, build_record, load_existing_hashes


def iter_markdown_files(directory: str | Path) -> list[Path]:
    """Return all .md files under directory (recursive)."""
    return sorted(Path(directory).rglob("*.md"))


def index_file(path: Path, known_hashes: set[str] | None = None) -> dict | None:
    """Embed a single markdown file and append it to the store.

    Returns the stored record, or None if the file's content is already known.
    """

    content = path.read_text(encoding="utf-8")
    text_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if known_hashes is not None and text_hash in known_hashes:
        return None

    record = build_record(str(path.resolve()), content)
    append_record(record)
    if known_hashes is not None:
        known_hashes.add(text_hash)

    return record


def index_directory(directory: str | Path) -> dict:
    """Embed and store every markdown file under directory (recursive).

    Skips files whose content is already present in the store. Returns a
    summary dict with indexed/skipped/total counts.
    """

    known = load_existing_hashes()
    indexed = skipped = 0
    for path in iter_markdown_files(directory):
        if index_file(path, known) is None:
            skipped += 1
        else:
            indexed += 1

    return {"indexed": indexed, "skipped": skipped, "total": indexed + skipped}


def main() -> None:
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    summary = index_directory(directory)

    print(
        f"Indexed {summary['indexed']}, skipped {summary['skipped']} "
        f"(of {summary['total']}) markdown files"
    )


if __name__ == "__main__":
    main()
