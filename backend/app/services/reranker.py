from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.config import Settings, get_settings


class Reranker:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self.settings.reranker_model)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[tuple[object, float]],
        top_k: int,
    ) -> list[tuple[object, float]]:
        if not candidates:
            return []

        pairs = [(query, doc.page_content) for doc, _ in candidates]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
        return [(doc, float(score)) for (doc, _), score in ranked[:top_k]]


@lru_cache
def get_reranker() -> Reranker:
    return Reranker(get_settings())
