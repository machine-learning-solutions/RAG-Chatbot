import logging

import httpx

from app.config import get_settings
from app.services.vector_store import get_vector_store_manager

logger = logging.getLogger(__name__)


async def warmup_models() -> None:
    """Pre-load embedding weights and Ollama so visitor requests stay under proxy limits."""
    settings = get_settings()

    try:
        manager = get_vector_store_manager()
        store = manager.store  # loads multilingual-e5-large into memory
        await store.embeddings.aembed_query("warmup")
        logger.info("Embedding model warmed up")
    except Exception as exc:
        logger.warning("Embedding warmup failed: %s", exc)

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{settings.ollama_base_url}/api/generate",
                json={
                    "model": settings.ollama_model,
                    "prompt": "warmup",
                    "stream": False,
                    "options": {"num_predict": 1, "think": False},
                },
            )
            response.raise_for_status()
        logger.info("Ollama model warmed up")
    except Exception as exc:
        logger.warning("Ollama warmup failed: %s", exc)
