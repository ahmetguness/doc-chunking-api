"""Application configuration module.

Defines embedding model configs, app defaults, environment-based settings,
and utility functions for model lookup and device detection.
"""

import os
from pydantic_settings import BaseSettings


EMBEDDING_MODELS: dict[str, dict] = {
    "all-MiniLM-L6-v2": {
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "max_length": 256,
        "language": "en",
        "description": "Fast English model, good balance of speed and quality",
        "dimension": 384,
        "query_prefix": "",
        "passage_prefix": "",
    },
    "paraphrase-multilingual-MiniLM-L12-v2": {
        "model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "max_length": 128,
        "language": "multi",
        "description": "Multilingual model supporting 50+ languages including Turkish",
        "dimension": 384,
        "query_prefix": "",
        "passage_prefix": "",
    },
    "all-mpnet-base-v2": {
        "model_id": "sentence-transformers/all-mpnet-base-v2",
        "max_length": 384,
        "language": "en",
        "description": "High quality English model, best overall performance",
        "dimension": 768,
        "query_prefix": "",
        "passage_prefix": "",
    },
    "bge-base-en-v1.5": {
        "model_id": "BAAI/bge-base-en-v1.5",
        "max_length": 512,
        "language": "en",
        "description": "BGE base English model with strong retrieval performance",
        "dimension": 768,
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "passage_prefix": "",
    },
    "bge-large-en-v1.5": {
        "model_id": "BAAI/bge-large-en-v1.5",
        "max_length": 512,
        "language": "en",
        "description": "BGE large English model, top-tier retrieval quality",
        "dimension": 1024,
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "passage_prefix": "",
    },
    "BAAI/bge-m3": {
        "model_id": "BAAI/bge-m3",
        "max_length": 8192,
        "language": "multi",
        "description": "BGE M3 multilingual model, supports 100+ languages",
        "dimension": 1024,
        "query_prefix": "",
        "passage_prefix": "",
    },
    "multilingual-e5-large": {
        "model_id": "intfloat/multilingual-e5-large",
        "max_length": 512,
        "language": "multi",
        "description": "E5 large multilingual model, strong cross-lingual retrieval",
        "dimension": 1024,
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
    "e5-base-v2": {
        "model_id": "intfloat/e5-base-v2",
        "max_length": 512,
        "language": "en",
        "description": "E5 base v2 English model, efficient retrieval performance",
        "dimension": 768,
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
    "e5-large-v2": {
        "model_id": "intfloat/e5-large-v2",
        "max_length": 512,
        "language": "en",
        "description": "E5 large v2 English model, high quality retrieval",
        "dimension": 1024,
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
}

APP_CONFIG: dict = {
    "title": "Document Chunking & Embedding API",
    "supported_formats": {".pdf", ".docx", ".txt", ".csv", ".xlsx", ".xls", ".zip"},
    "max_files": int(os.environ.get("MAX_FILES", 10)),
    "default_values": {
        "max_tokens": int(os.environ.get("MAX_TOKENS", 512)),
        "overlap": int(os.environ.get("OVERLAP", 100)),
        "values_only_threshold": 2,
        "min_cols_for_table": 2,
        "min_rows_for_table": 2,
        "include_column_names": True,
        "attach_row_data": True,
        "flatten_row_values_to_root": True,
        "output_text_column": "",
        "attach_context_on_split": True,
    },
}


def get_model_config(model_name: str) -> dict:
    """Return the configuration dict for the given model name.

    If the model is present in the built-in EMBEDDING_MODELS registry, its
    rich config (prefixes, max_length, dimension) is returned. Otherwise a
    permissive default config is returned so that user-provided models
    (validated at deploy time against OpenRouter / HuggingFace) still work.
    """
    if model_name in EMBEDDING_MODELS:
        return EMBEDDING_MODELS[model_name]
    return {
        "model_id": model_name,
        "max_length": 512,
        "language": "unknown",
        "description": "User-provided model (not in built-in registry)",
        "dimension": None,
        "query_prefix": "",
        "passage_prefix": "",
    }


class Settings(BaseSettings):
    """Environment-based application settings."""

    MAX_TABLE_ROWS: int = 500_000
    MAX_FILE_SIZE_MB: int = 100
    MAX_CONCURRENT_REQUESTS: int = 2
    MAX_FILES: int = 10
    PRELOAD_MODELS: str = ""
    LOG_LEVEL: str = "INFO"
    API_PORT: int = 8000
    REQUEST_TIMEOUT_SECONDS: int = 1800
    CONCURRENCY_ACQUIRE_TIMEOUT: int = 30

    # ------------------------------------------------------------------
    # Embedding backend selection (chosen at deploy time via setup wizard)
    # ------------------------------------------------------------------
    # EMBEDDING_MODE: "local" (run model on this service) or "cloud"
    # (use a remote OpenAI-compatible embeddings API such as OpenRouter).
    EMBEDDING_MODE: str = "local"
    # EMBEDDING_DEVICE: "auto" | "cpu" | "gpu"/"cuda" (only used in local mode).
    EMBEDDING_DEVICE: str = "auto"
    # Active model for local mode (HuggingFace repo id or registry key).
    MODEL_NAME: str = ""
    # Tokenizer used for chunking. In cloud mode this is the HuggingFace repo
    # whose tokenizer matches the remote embedding model. Falls back to
    # MODEL_NAME / EMBEDDING_API_MODEL when empty.
    TOKENIZER_ID: str = ""

    # Cloud mode: OpenAI-compatible embeddings API (OpenRouter by default).
    EMBEDDING_API_BASE_URL: str = "https://openrouter.ai/api/v1"
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_API_MODEL: str = ""
    EMBEDDING_API_TIMEOUT: int = 60

    # HuggingFace token for gated/private tokenizers or models.
    HF_TOKEN: str = ""

    model_config = {"env_prefix": ""}


settings = Settings()


def get_device(prefer: str | None = None) -> str:
    """Resolve the torch device string based on EMBEDDING_DEVICE.

    Args:
        prefer: Optional override for the configured device. Accepts
            "auto", "cpu", "gpu", or "cuda".

    Returns:
        "cuda" or "cpu".

    Raises:
        RuntimeError: If "cuda"/"gpu" is explicitly requested but no CUDA
            device (or PyTorch) is available.
    """
    mode = (prefer or settings.EMBEDDING_DEVICE or "auto").strip().lower()
    if mode == "gpu":
        mode = "cuda"

    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        if mode == "cuda":
            raise RuntimeError(
                "GPU istendi ancak PyTorch yüklü değil. CUDA destekli bir "
                "ortamda çalıştırın veya EMBEDDING_DEVICE=cpu kullanın."
            )
        return "cpu"

    if mode == "cpu":
        return "cpu"
    if mode == "cuda":
        if not cuda_available:
            raise RuntimeError(
                "GPU istendi ancak CUDA bulunamadı. EMBEDDING_DEVICE=cpu "
                "kullanın veya CUDA destekli bir ortamda çalıştırın."
            )
        return "cuda"
    # auto
    return "cuda" if cuda_available else "cpu"
