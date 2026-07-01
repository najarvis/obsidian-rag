import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from obsidian_rag.embeddings import get_model_name
from obsidian_rag.indexer import index_directory as index_directory_files
from obsidian_rag.storage import append_record, build_record, get_store_path

mcp = FastMCP("obsidian-rag")


def resolve_text(reference: str, text: str | None) -> str:
    if text is not None:
        return text

    path = Path(reference)
    if not path.is_file():
        raise ValueError(f"Reference is not a readable file: {reference}")
    return path.read_text(encoding="utf-8")


@mcp.tool()
def embed_and_store(reference: str, text: str | None = None) -> str:
    """Generate an embedding for text and store it with a reference.

    Args:
        reference: Absolute file path or arbitrary reference string identifying the source.
        text: Inline text to embed. If omitted, reference is read as a file path.
    """
    content = resolve_text(reference, text)
    record = build_record(reference, content)
    append_record(record)

    return json.dumps(
        {
            "id": record["id"],
            "reference": reference,
            "dimensions": len(record["embedding"]),
            "model": record["model"],
            "store_path": str(get_store_path().resolve()),
        }
    )


@mcp.tool()
def index_directory(directory: str) -> str:
    """Embed and store every markdown (.md) file under a directory (recursive).

    Files whose content is already present in the store are skipped.

    Args:
        directory: Absolute path to the directory to index.
    """
    path = Path(directory)
    if not path.is_dir():
        raise ValueError(f"Not a directory: {directory}")
    summary = index_directory_files(path)
    return json.dumps(
        {
            "directory": str(path.resolve()),
            "indexed": summary["indexed"],
            "skipped": summary["skipped"],
            "total": summary["total"],
            "model": get_model_name(),
            "store_path": str(get_store_path().resolve()),
        }
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
