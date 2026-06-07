"""API router for document chunking and embedding endpoints.

Provides POST /process, GET /models, and GET /health endpoints.
"""

import asyncio
import logging
import time
from typing import Literal, Optional

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse

from app.config import APP_CONFIG, EMBEDDING_MODELS, settings
from app.middleware.timeout import RequestTimeoutError, process_with_timeout
from app.schemas.response import (
    ErrorResponse,
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    ProcessResponse,
)
from app.services.api_embedding import ApiEmbeddingEngine
from app.services.chunker import Chunker
from app.services.embedding import EmbeddingEngine
from app.services.file_processor import FileProcessor
from app.services.model_manager import ModelManager
from app.services.normalizer import TextNormalizer
from app.services.response_formatter import ResponseFormatter
from app.services.table_processor import TableProcessor
from app.api.dependencies import ConcurrencyLimiter, verify_chunking_token

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults from config (NOT hardcoded)
# ---------------------------------------------------------------------------
_defaults = APP_CONFIG["default_values"]

# ---------------------------------------------------------------------------
# Embedding backend selection (fixed at deploy time via EMBEDDING_MODE)
# ---------------------------------------------------------------------------
_EMBEDDING_MODE = (settings.EMBEDDING_MODE or "local").strip().lower()
_IS_CLOUD = _EMBEDDING_MODE == "cloud"


def _default_model_name() -> str:
    """Resolve the active embedding model name for this deployment."""
    if _IS_CLOUD:
        return settings.EMBEDDING_API_MODEL or next(iter(EMBEDDING_MODELS))
    return settings.MODEL_NAME or next(iter(EMBEDDING_MODELS))


def _tokenizer_id_for(model_name: str) -> str:
    """Resolve which tokenizer to use for chunk sizing."""
    return settings.TOKENIZER_ID or model_name


# ---------------------------------------------------------------------------
# Module-level service instances
# ---------------------------------------------------------------------------
model_manager = ModelManager()
if _IS_CLOUD:
    embedding_engine = ApiEmbeddingEngine()
    logger.info(
        "Embedding backend: CLOUD (model=%s, base_url=%s)",
        settings.EMBEDDING_API_MODEL, settings.EMBEDDING_API_BASE_URL,
    )
else:
    embedding_engine = EmbeddingEngine(model_manager)
    logger.info(
        "Embedding backend: LOCAL (model=%s, device=%s)",
        _default_model_name(), settings.EMBEDDING_DEVICE,
    )
file_processor = FileProcessor()
table_processor = TableProcessor()
text_normalizer = TextNormalizer()
chunker = Chunker()
response_formatter = ResponseFormatter()
concurrency_limiter = ConcurrencyLimiter(
    max_concurrent=settings.MAX_CONCURRENT_REQUESTS,
    acquire_timeout=settings.CONCURRENCY_ACQUIRE_TIMEOUT,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _validate_file_size(file: UploadFile, max_bytes: int) -> bytes:
    """Read file content and validate size."""
    content = await file.read()
    if len(content) > max_bytes:
        max_mb = max_bytes // (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"Dosya '{file.filename}' boyutu limiti aşıyor: {max_mb}MB",
        )
    return content


async def _process_single_file(
    file_bytes: bytes,
    filename: str,
    *,
    model_name: str,
    normalization: str,
    max_tokens: int,
    overlap: int,
    skip_embedding: bool,
    embedding_batch_size: int,
    prefix_mode: str,
    zip_password: Optional[str],
    include_column_names: bool,
    attach_row_data: bool,
    flatten_row_values_to_root: bool,
    output_text_column: str,
    values_only_threshold: int,
    min_cols_for_table: Optional[int],
    min_rows_for_table: Optional[int],
    attach_context_on_split: bool,
) -> dict:
    """Run the full pipeline for a single file and return a result dict."""
    # Step 1: Parse file  list[FileContent]
    file_contents = file_processor.process(file_bytes, filename, zip_password)

    # Step 2: Get tokenizer for chunking / table processing.
    # In cloud mode this loads ONLY the tokenizer (no model weights).
    tokenizer = await model_manager.get_tokenizer(_tokenizer_id_for(model_name))

    all_chunks = []
    for fc in file_contents:
        if fc.file_type == "table" and fc.dataframe is not None:
            chunks = table_processor.process(
                df=fc.dataframe,
                filename=fc.filename,
                file_extension=fc.file_extension,
                tokenizer=tokenizer,
                include_column_names=include_column_names,
                attach_row_data=attach_row_data,
                flatten_row_values_to_root=flatten_row_values_to_root,
                output_text_column=output_text_column,
                values_only_threshold=values_only_threshold,
                min_cols_for_table=min_cols_for_table,
                min_rows_for_table=min_rows_for_table,
                max_tokens=max_tokens,
                overlap=overlap,
                normalization=normalization,
                attach_context_on_split=attach_context_on_split,
            )
        else:
            # Text file: normalize then chunk
            normalized = text_normalizer.normalize(fc.content, mode=normalization)
            chunks = chunker.chunk(
                text=normalized,
                tokenizer=tokenizer,
                max_tokens=max_tokens,
                overlap=overlap,
                filename=fc.filename,
            )
        all_chunks.extend(chunks)

    # Step 3: Embedding (with concurrency limiter)
    embeddings = None
    if not skip_embedding:
        # Set embedding_model_id on all chunks before embedding
        for chunk in all_chunks:
            chunk.embedding_model_id = model_name
        async with concurrency_limiter.throttle():
            embeddings = await embedding_engine.generate(
                all_chunks, model_name, embedding_batch_size, prefix_mode=prefix_mode
            )

    return {
        "filename": filename,
        "chunks": all_chunks,
        "embeddings": embeddings,
    }


# ---------------------------------------------------------------------------
# POST /process
# ---------------------------------------------------------------------------

@router.post(
    "/process",
    response_model=None,
    dependencies=[Depends(verify_chunking_token)],
    responses={
        200: {
            "content": {
                "application/json": {},
                "application/zip": {},
            },
        },
    },
)
async def process_documents(
    files: list[UploadFile],
    model_name: Optional[str] = Form(default=None),
    normalization: Literal["none", "lowercase", "uppercase"] = Form(default="none"),
    max_tokens: int = Form(default=_defaults["max_tokens"]),
    overlap: int = Form(default=_defaults["overlap"]),
    skip_embedding: bool = Form(default=False),
    response_format: Literal["json", "json_with_embeddings", "zip"] = Form(default="json"),
    embedding_batch_size: int = Form(default=32),
    prefix_mode: Literal["passage", "query"] = Form(default="passage"),
    zip_password: Optional[str] = Form(default=None),
    include_column_names: bool = Form(default=_defaults["include_column_names"]),
    attach_row_data: bool = Form(default=_defaults["attach_row_data"]),
    flatten_row_values_to_root: bool = Form(default=_defaults["flatten_row_values_to_root"]),
    output_text_column: str = Form(default=_defaults["output_text_column"]),
    values_only_threshold: int = Form(default=_defaults["values_only_threshold"]),
    min_cols_for_table: Optional[int] = Form(default=_defaults.get("min_cols_for_table")),
    min_rows_for_table: Optional[int] = Form(default=_defaults.get("min_rows_for_table")),
    attach_context_on_split: bool = Form(default=_defaults.get("attach_context_on_split", True)),
) -> JSONResponse | StreamingResponse:
    """Process uploaded documents: read  normalize  chunk  embed  format."""

    # MAX_FILES check
    if len(files) > settings.MAX_FILES:
        raise HTTPException(
            status_code=422,
            detail=f"Maksimum {settings.MAX_FILES} dosya yüklenebilir",
        )

    # Default model name: the deployment's active model.
    if model_name is None:
        model_name = _default_model_name()
    # Models are validated at deploy time (against OpenRouter in cloud mode,
    # HuggingFace in local mode), so request-time names are accepted as-is.

    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    start_time = time.time()

    async def _pipeline() -> JSONResponse | StreamingResponse:
        file_results: list[dict] = []

        for upload_file in files:
            try:
                content = await _validate_file_size(upload_file, max_bytes)
                result = await _process_single_file(
                    content,
                    upload_file.filename or "unknown",
                    model_name=model_name,
                    normalization=normalization,
                    max_tokens=max_tokens,
                    overlap=overlap,
                    skip_embedding=skip_embedding,
                    embedding_batch_size=embedding_batch_size,
                    prefix_mode=prefix_mode,
                    zip_password=zip_password,
                    include_column_names=include_column_names,
                    attach_row_data=attach_row_data,
                    flatten_row_values_to_root=flatten_row_values_to_root,
                    output_text_column=output_text_column,
                    values_only_threshold=values_only_threshold,
                    min_cols_for_table=min_cols_for_table,
                    min_rows_for_table=min_rows_for_table,
                    attach_context_on_split=attach_context_on_split,
                )
                file_results.append(result)
            except HTTPException:
                # Re-raise HTTP exceptions (e.g. file size) without partial failure
                raise
            except asyncio.TimeoutError:
                # ConcurrencyLimiter timeout  429
                raise
            except Exception as exc:
                # Partial failure: record error, continue with other files
                logger.warning(
                    "Error processing file '%s': %s",
                    upload_file.filename,
                    exc,
                )
                file_results.append(
                    {
                        "filename": upload_file.filename or "unknown",
                        "chunks": [],
                        "embeddings": None,
                        "error": ErrorResponse(
                            error_code=getattr(exc, "error_code", "PROCESSING_ERROR"),
                            message=str(exc),
                        ),
                    }
                )

        processing_time = time.time() - start_time
        effective_model = model_name if not skip_embedding else None

        if response_format == "zip":
            return response_formatter.format_zip(
                file_results, effective_model, processing_time
            )
        elif response_format == "json_with_embeddings":
            resp = response_formatter.format_json_with_embeddings(
                file_results, effective_model, processing_time
            )
            return JSONResponse(content=resp.model_dump())
        else:
            resp = response_formatter.format_json(
                file_results, effective_model, processing_time
            )
            return JSONResponse(content=resp.model_dump())

    # Wrap entire pipeline with timeout
    try:
        return await process_with_timeout(
            _pipeline(), timeout_seconds=settings.REQUEST_TIMEOUT_SECONDS
        )
    except RequestTimeoutError:
        return JSONResponse(
            status_code=408,
            content={
                "error_code": "REQUEST_TIMEOUT",
                "message": "İstek zaman aşımına uğradı",
                "detail": f"Maksimum işleme süresi: {settings.REQUEST_TIMEOUT_SECONDS}s",
            },
        )
    except asyncio.TimeoutError:
        # ConcurrencyLimiter acquire timeout  429
        return JSONResponse(
            status_code=429,
            content={
                "error_code": "TOO_MANY_REQUESTS",
                "message": "Sunucu meşgul, lütfen daha sonra tekrar deneyin",
                "detail": f"Eşzamanlı istek limiti: {settings.MAX_CONCURRENT_REQUESTS}",
            },
        )


# ---------------------------------------------------------------------------
# GET /models
# ---------------------------------------------------------------------------

@router.get("/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    """Return the list of supported embedding models."""
    default_model = next(iter(EMBEDDING_MODELS))
    models = [
        ModelInfo(
            name=name,
            dimension=cfg["dimension"],
            max_seq_length=cfg["max_length"],
            language=cfg["language"],
            description=cfg.get("description"),
        )
        for name, cfg in EMBEDDING_MODELS.items()
    ]
    return ModelsResponse(models=models, default_model=default_model)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return health status and concurrency limiter stats."""
    stats = concurrency_limiter.stats
    loaded = model_manager.loaded_model_names()
    status = "healthy" if stats["active"] < settings.MAX_CONCURRENT_REQUESTS else "degraded"
    return HealthResponse(
        status=status,
        active_requests=stats["active"],
        waiting_requests=stats["waiting"],
        loaded_models=loaded,
    )
