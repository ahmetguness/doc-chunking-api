"""Model management service for loading, caching, and serving embedding models.

Manages SentenceTransformer models and AutoTokenizer instances with
async-safe caching and lazy/preload strategies.
"""

import asyncio
import logging

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from app.config import EMBEDDING_MODELS, get_device, get_model_config

logger = logging.getLogger(__name__)


class ModelManager:
    """Loads, caches, and serves SentenceTransformer + AutoTokenizer pairs.

    Thread-safe via asyncio.Lock. Validates model names against
    EMBEDDING_MODELS config before loading.
    """

    def __init__(self) -> None:
        self._cache: dict[str, tuple[SentenceTransformer, AutoTokenizer]] = {}
        self._lock = asyncio.Lock()

    async def get_model(
        self, model_name: str
    ) -> tuple[SentenceTransformer, AutoTokenizer]:
        config = get_model_config(model_name)
        model_id = config["model_id"]

        async with self._lock:
            if model_name not in self._cache:
                device = get_device()
                logger.info(
                    "Loading model '%s' (model_id=%s) on device=%s",
                    model_name, model_id, device,
                )
                model = await asyncio.to_thread(
                    SentenceTransformer, model_id, device=device
                )
                tokenizer = await asyncio.to_thread(
                    AutoTokenizer.from_pretrained, model_id
                )
                self._cache[model_name] = (model, tokenizer)
                logger.info("Model '%s' loaded and cached.", model_name)
            return self._cache[model_name]

    async def get_tokenizer(self, model_name: str) -> AutoTokenizer:
        """Return only the AutoTokenizer for the given model.

        Loads the full model+tokenizer pair if not already cached.

        Args:
            model_name: Key in EMBEDDING_MODELS config.

        Returns:
            AutoTokenizer instance.

        Raises:
            ValueError: If model_name is not in EMBEDDING_MODELS.
        """
        _, tokenizer = await self.get_model(model_name)
        return tokenizer

    async def preload(self, model_names: list[str]) -> None:
        """Preload models at startup (warm-up).

        Args:
            model_names: List of model name keys to preload.
        """
        for name in model_names:
            logger.info("Preloading model '%s'...", name)
            await self.get_model(name)

    def loaded_model_names(self) -> list[str]:
        """Return the list of currently cached model names.

        Provides public access to cache keys without exposing the
        private _cache attribute directly.
        """
        return list(self._cache.keys())

    def list_models(self) -> list[dict]:
        """Return the list of supported models from EMBEDDING_MODELS config.

        Returns:
            List of dicts with name, model_id, dimension, max_length,
            language, and description for each supported model.
        """
        return [
            {
                "name": name,
                "model_id": cfg["model_id"],
                "dimension": cfg["dimension"],
                "max_length": cfg["max_length"],
                "language": cfg["language"],
                "description": cfg["description"],
            }
            for name, cfg in EMBEDDING_MODELS.items()
        ]
