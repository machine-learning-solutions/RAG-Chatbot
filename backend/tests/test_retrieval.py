from langchain_core.documents import Document

from app.services.hybrid_search import reciprocal_rank_fusion


def _doc(doc_id: str, text: str) -> Document:
    doc = Document(page_content=text)
    doc.id = doc_id
    return doc


def test_rrf_promotes_chunks_ranked_high_in_multiple_lists() -> None:
    list_a = [
        (_doc("a", "alpha"), 0.9),
        (_doc("b", "beta"), 0.8),
        (_doc("c", "gamma"), 0.7),
    ]
    list_b = [
        (_doc("b", "beta"), 12.0),
        (_doc("d", "delta"), 11.0),
        (_doc("a", "alpha"), 10.0),
    ]

    fused = reciprocal_rank_fusion([list_a, list_b], k=3, rrf_constant=60)

    assert [doc.id for doc, _ in fused] == ["b", "a", "c"]


def test_rrf_handles_single_list() -> None:
    ranked = [(_doc("x", "one"), 1.0), (_doc("y", "two"), 0.5)]
    fused = reciprocal_rank_fusion([ranked], k=2)
    assert [doc.id for doc, _ in fused] == ["x", "y"]
