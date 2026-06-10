from functools import lru_cache
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from turbovec.langchain import TurboQuantVectorStore

from app.config import Settings, get_settings
from app.services.torch_device import get_inference_device


class E5Embeddings(Embeddings):
    """multilingual-e5-large expects query:/passage: prefixes."""

    def __init__(self, model_name: str) -> None:
        self._model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": get_inference_device()},
            encode_kwargs={"normalize_embeddings": True},
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.embed_documents([f"passage: {t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._model.embed_query(f"query: {text}")

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._model.aembed_documents([f"passage: {t}" for t in texts])

    async def aembed_query(self, text: str) -> list[float]:
        return await self._model.aembed_query(f"query: {text}")


def vector_store_stats(settings: Settings | None = None) -> dict:
    """Lightweight stats without loading the embedding model."""
    cfg = settings or get_settings()
    store_path = Path(cfg.vector_store_path)
    return {
        "path": str(store_path),
        "indexed": (store_path / "index.tvim").exists(),
        "bit_width": cfg.turbovec_bit_width,
        "embedding_model": cfg.embedding_model,
        "docstore_exists": (store_path / "docstore.json").exists(),
    }


class VectorStoreManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store_path = Path(settings.vector_store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self._embeddings: Embeddings | None = None
        self._store: TurboQuantVectorStore | None = None

    def _ensure_loaded(self) -> None:
        if self._store is not None:
            return
        self._embeddings = E5Embeddings(self.settings.embedding_model)
        index_file = self.store_path / "index.tvim"
        if index_file.exists():
            self._store = TurboQuantVectorStore.load(
                str(self.store_path),
                embedding=self._embeddings,
            )
        else:
            self._store = TurboQuantVectorStore(
                embedding=self._embeddings,
                bit_width=self.settings.turbovec_bit_width,
            )

    @property
    def store(self) -> TurboQuantVectorStore:
        self._ensure_loaded()
        assert self._store is not None
        return self._store

    def add_documents(self, documents: list) -> list[str]:
        ids = [doc.id for doc in documents]
        self.store.add_documents(documents, ids=ids)
        self.persist()
        return ids

    def delete_by_ids(self, ids: list[str]) -> None:
        if ids:
            self.store.delete(ids)
            self.persist()

    def persist(self) -> None:
        self.store.dump(str(self.store_path))

    def search(
        self,
        query: str,
        k: int,
        document_id: str | None = None,
    ) -> list:
        search_kwargs: dict = {"k": k}
        if document_id:
            search_kwargs["filter"] = {"document_id": document_id}
        return self.store.similarity_search_with_score(query, **search_kwargs)

    def stats(self) -> dict:
        return vector_store_stats(self.settings)


@lru_cache
def get_vector_store_manager() -> VectorStoreManager:
    return VectorStoreManager(get_settings())
