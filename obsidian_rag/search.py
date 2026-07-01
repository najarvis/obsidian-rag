from obsidian_rag.embeddings import embed_text
from obsidian_rag.storage import load_records


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5

    # An empty document produces a zero vector, for which cosine is undefined.
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def find_related(
    text: str, top_n: int = 5, exclude_hash: str | None = None
) -> list[dict]:
    query = embed_text(text)

    scored: list[tuple[float, dict]] = []
    for record in load_records():
        embedding = record.get("embedding")

        # Skip records with missing or mismatched-dimension vectors, e.g. those
        # written by a different embedding model or dimensionality.
        if not embedding or len(embedding) != len(query):
            continue

        if exclude_hash is not None and record.get("text_hash") == exclude_hash:
            continue

        scored.append((cosine_similarity(query, embedding), record))

    scored.sort(key=lambda item: item[0], reverse=True)

    return [
        {
            "reference": record.get("reference"),
            "score": round(score, 6),
            "text_preview": record.get("text_preview"),
            "id": record.get("id"),
        }
        for score, record in scored[:top_n]
    ]
