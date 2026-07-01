import hashlib

import pytest

from obsidian_rag.search import cosine_similarity, find_related
from obsidian_rag.storage import append_record, build_record


def test_cosine_similarity_identical_vectors():
    vec = [1.0, 2.0, 3.0]
    assert cosine_similarity(vec, vec) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_find_related_empty_store():
    assert find_related("anything") == []


def test_find_related_ranks_by_similarity():
    append_record(build_record("cats", "cats and kittens love to nap in the sun"))
    append_record(build_record("dogs", "dogs and puppies enjoy long walks outside"))
    append_record(build_record("cars", "engines pistons torque and horsepower"))

    results = find_related("kittens napping in the sun", top_n=3)

    assert results[0]["reference"] == "cats"
    assert len(results) == 3
    # Scores must be sorted in descending order.
    scores = [item["score"] for item in results]
    assert scores == sorted(scores, reverse=True)


def test_find_related_respects_top_n():
    for index in range(5):
        append_record(build_record(f"ref-{index}", f"document number {index}"))

    assert len(find_related("document", top_n=2)) == 2


def test_find_related_excludes_hash():
    content = "unique note about mountains"
    append_record(build_record("self", content))
    append_record(build_record("other", "a different note about oceans"))

    exclude = hashlib.sha256(content.encode("utf-8")).hexdigest()
    results = find_related(content, exclude_hash=exclude)

    references = [item["reference"] for item in results]
    assert "self" not in references
