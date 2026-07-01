import hashlib
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

from obsidian_rag.embeddings import embed_text, get_model_name


def get_store_path() -> Path:
    return Path(os.getenv("EMBEDDINGS_STORE_PATH", "data/embeddings.jsonl"))


def build_record(reference: str, content: str) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "reference": reference,
        "text_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "text_preview": content[:200],
        "embedding": embed_text(content),
        "model": get_model_name(),
        "created_at": datetime.now(UTC).isoformat(),
    }


def append_record(record: dict) -> None:
    path = get_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


def load_existing_hashes() -> set[str]:
    path = get_store_path()
    if not path.is_file():
        return set()

    hashes: set[str] = set()
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                hashes.add(json.loads(line)["text_hash"])

            except (json.JSONDecodeError, KeyError):
                continue

    return hashes


def load_records() -> list[dict]:
    path = get_store_path()
    if not path.is_file():
        return []

    records: list[dict] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))

            except json.JSONDecodeError:
                continue

    return records
