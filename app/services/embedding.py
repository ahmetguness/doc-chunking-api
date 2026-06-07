"""Embedding engine service for generating vector embeddings from text chunks.

Uses SentenceTransformer models via ModelManager to produce embeddings
in batches, returning results as a numpy ndarray.  Supports instruction
prefix injection required by BGE and E5 model families.
"""

import asyncio
import logging
from typing import Literal

import numpy as np

from app.config import get_model_config
from app.schemas.internal import Chunk
from app.services.model_manager import ModelManager

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Generates embedding vectors from text chunks using sentence-transformers.

    Delegates model loading/caching to ModelManager and processes chunks
    in configurable batch sizes.  Automatically prepends the appropriate
    prefix (query_prefix / passage_prefix) defined in the model config.
    """

    def __init__(self, model_manager: ModelManager) -> None:
        self.model_manager = model_manager

    async def generate(
        self,
        chunks: list[Chunk],
        model_name: str,
        batch_size: int = 32,
        prefix_mode: Literal["passage", "query"] = "passage",
    ) -> np.ndarray:
        """Generate embeddings for the given chunks.

        Args:
            chunks: List of Chunk objects to embed.
            model_name: Key in EMBEDDING_MODELS config.
            batch_size: Number of chunks to encode per batch.
            prefix_mode: Which prefix to prepend - ``"passage"`` for
                document indexing (default) or ``"query"`` for search
                queries.  Only affects models that define non-empty
                prefixes (e.g. BGE, E5).

        Returns:
            numpy ndarray of shape (num_chunks, embedding_dim).
            Returns empty array with shape (0,) when chunks list is empty.
        """
        if not chunks:
            return np.array([])

        model, _tokenizer = await self.model_manager.get_model(model_name)

        # Resolve prefix from model config
        config = get_model_config(model_name)
        if prefix_mode == "query":
            prefix = config.get("query_prefix", "")
        else:
            prefix = config.get("passage_prefix", "")

        texts = [prefix + chunk.text for chunk in chunks]

        # Hard guard: max_length aşan metinleri truncate et
        max_length = config.get("max_length", 8192)
        truncated_count = 0
        for i, text in enumerate(texts):
            tokens = _tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) > max_length:
                truncated_count += 1
                texts[i] = _tokenizer.decode(tokens[:max_length], skip_special_tokens=True)

        if truncated_count:
            logger.warning(
                "%d/%d chunk max_length (%d) aşıyordu, truncate edildi",
                truncated_count, len(texts), max_length,
            )

        logger.info(
            "Generating embeddings for %d chunks with model '%s' "
            "(batch_size=%d, prefix_mode=%s, prefix=%r)",
            len(texts),
            model_name,
            batch_size,
            prefix_mode,
            prefix or "(none)",
        )

        embeddings = await asyncio.to_thread(
            model.encode,
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        result = np.array(embeddings)

        logger.info(
            "Embeddings generated: shape=%s, dtype=%s", result.shape, result.dtype
        )

        return result
