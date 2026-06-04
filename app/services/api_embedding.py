"""Cloud embedding provider using an OpenAI-compatible embeddings API.

Works with any provider that implements the OpenAI ``/embeddings`` contract,
including OpenRouter (``https://openrouter.ai/api/v1``). The remote service
computes the embedding vectors; this module only batches text, calls the
endpoint, and normalizes the returned vectors.
"""

import asyncio
import logging
from typing import Literal

import httpx
import numpy as np

from app.config import get_model_config, settings
from app.schemas.internal import Chunk

logger = logging.getLogger(__name__)


class ApiEmbeddingEngine:
    """Generates embeddings by calling a remote OpenAI-compatible API.

    The chunk text is sent to ``{base_url}/embeddings``. Vectors are
    L2-normalized locally to match the local SentenceTransformer behaviour
    (``normalize_embeddings=True``).
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self.base_url = (base_url or settings.EMBEDDING_API_BASE_URL).rstrip("/")
        self.api_key = api_key or settings.EMBEDDING_API_KEY
        self.model = model or settings.EMBEDDING_API_MODEL
        self.timeout = timeout or settings.EMBEDDING_API_TIMEOUT

    async def generate(
        self,
        chunks: list[Chunk],
        model_name: str,
        batch_size: int = 32,
        prefix_mode: Literal["passage", "query"] = "passage",
    ) -> np.ndarray:
        """Generate embeddings for *chunks* via the remote API.

        Signature matches the local EmbeddingEngine so callers are agnostic
        to the backend.
        """
        if not chunks:
            return np.array([])

        # Resolve instruction prefix (BGE/E5 families) from registry config.
        config = get_model_config(model_name)
        if prefix_mode == "query":
            prefix = config.get("query_prefix", "")
        else:
            prefix = config.get("passage_prefix", "")

        texts = [prefix + chunk.text for chunk in chunks]

        logger.info(
            "Requesting embeddings from API for %d chunks "
            "(model=%s, base_url=%s, batch_size=%d, prefix_mode=%s)",
            len(texts), self.model, self.base_url, batch_size, prefix_mode,
        )

        vectors: list[list[float]] = []
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                payload = {"model": self.model, "input": batch}
                resp = await client.post(
                    f"{self.base_url}/embeddings",
                    headers=headers,
                    json=payload,
                )
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"Embedding API hatası ({resp.status_code}): {resp.text}"
                    )
                data = resp.json()
                # OpenAI contract: {"data": [{"embedding": [...], "index": i}, ...]}
                items = sorted(data["data"], key=lambda d: d.get("index", 0))
                vectors.extend(item["embedding"] for item in items)

        result = np.array(vectors, dtype=np.float32)

        # L2-normalize to match local normalize_embeddings=True behaviour.
        if result.size > 0:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            result = result / norms

        logger.info(
            "API embeddings received: shape=%s, dtype=%s", result.shape, result.dtype
        )
        return result
