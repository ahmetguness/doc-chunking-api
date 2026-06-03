"""Model management service for loading, caching, and serving embedding models.

Manages SentenceTransformer models and AutoTokenizer instances with
async-safe caching and lazy/preload strategies.

Two loading paths are supported:

* ``get_model`` loads a full SentenceTransformer (+ tokenizer) for **local**
  embedding mode. The device is resolved through :func:`app.config.get_device`.
* ``get_tokenizer`` loads **only** an ``AutoTokenizer``. This is what
  **cloud** mode uses: chunking needs a tokenizer to size chunks, but the
  heavy model weights live on the remote API, so we never download them.
"""

import asyncio
import logging

from transformers import AutoTokenizer

from app.config import EMBEDDING_MODELS, get_device, get_model_config, settings

logger = logging.getLogger(__name__)


class ModelManager:
    """Loads, caches, and serves SentenceTransformer and/or AutoTokenizer.

    Thread-safe via asyncio.Lock. Embedding models are only loaded in local
    mode; tokenizers can be loaded independently (used by cloud mode for
    chunk sizing).
    """

    def __init__(self) -> None:
        # Full model cache: name -> (SentenceTransformer, AutoTokenizer)
        self._cache: dict[str, tuple] = {}
        # Tokenizer-only cache: tokenizer_id -> AutoTokenizer
        self._tokenizer_cache: dict[str, object] = {}
        self._lock = asyncio.Lock()

    def _resolve_model_id(self, model_name: str) -> str:
        """Map a registry key or raw id to the underlying HuggingFace id."""
        config = get_model_config(model_name)
        return config.get("model_id", model_name)

    async def get_model(self, model_name: str) -> tuple:
        """Load (or fetch from cache) a full SentenceTransformer + tokenizer.

        Only used in local embedding mode.
        """
        # Imported lazily so cloud-only deployments need not import torch.
        from sentence_transformers import SentenceTransformer

        model_id = self._resolve_model_id(model_name)

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
                    self._load_tokenizer, model_id
                )
                self._cache[model_name] = (model, tokenizer)
                # Also populate the tokenizer-only cache.
                self._tokenizer_cache[model_id] = tokenizer
                logger.info("Model '%s' loaded and cached.", model_name)
            return self._cache[model_name]

    @staticmethod
    def _load_tokenizer(tokenizer_id: str):
        """Load an AutoTokenizer, passing an HF token when configured."""
        kwargs = {}
        if settings.HF_TOKEN:
            kwargs["token"] = settings.HF_TOKEN
        return AutoTokenizer.from_pretrained(tokenizer_id, **kwargs)

    async def get_tokenizer(self, tokenizer_id: str):
        """Return an AutoTokenizer for *tokenizer_id*, loading only the tokenizer.

        In local mode, if the full model for this id is already cached we
        reuse its tokenizer. Otherwise (e.g. cloud mode) only the lightweight
        tokenizer files are downloaded — never the model weights.

        Args:
            tokenizer_id: HuggingFace repo id or built-in registry key.

        Returns:
            AutoTokenizer instance.
        """
        resolved = self._resolve_model_id(tokenizer_id)

        async with self._lock:
            if resolved in self._tokenizer_cache:
                return self._tokenizer_cache[resolved]
            logger.info("Loading tokenizer only: '%s'", resolved)
            tokenizer = await asyncio.to_thread(self._load_tokenizer, resolved)
            self._tokenizer_cache[resolved] = tokenizer
            logger.info("Tokenizer '%s' loaded and cached.", resolved)
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
        """Return the list of currently cached model names."""
        names = list(self._cache.keys())
        # Surface tokenizer-only loads (cloud mode) too.
        for tid in self._tokenizer_cache:
            if tid not in names:
                names.append(f"tokenizer:{tid}")
        return names

    def list_models(self) -> list[dict]:
        """Return the list of supported models from EMBEDDING_MODELS config."""
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
