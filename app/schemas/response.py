"""Response schemas for the Document Chunking & Embedding API."""

from typing import Literal, Optional

from pydantic import BaseModel


class ChunkMetadata(BaseModel):
    """Metadata for a single chunk."""

    source_file: str
    chunk_index: int
    token_count: int
    is_tabular: bool = False
    table_type: Optional[str] = None  # "structured" or "unstructured"
    single_row: bool = False
    row_split: bool = False
    sentence_aware: bool = False
    table_columns: Optional[list[str]] = None
    embedding_model_id: Optional[str] = None
    row_data: Optional[dict] = None
    extra_metadata: Optional[dict] = None


class ChunkResult(BaseModel):
    """A single chunk with its text and metadata."""

    text: str
    metadata: ChunkMetadata


class ErrorResponse(BaseModel):
    """Standard error response format."""

    error_code: str
    message: str
    detail: Optional[str] = None


class FileResult(BaseModel):
    """Processing result for a single file."""

    filename: str
    chunks: list[ChunkResult]
    embeddings: Optional[list[list[float]]] = None
    embeddings_base64_npy: Optional[str] = None
    error: Optional[ErrorResponse] = None


class ProcessResponse(BaseModel):
    """Response for the /process endpoint."""

    results: list[FileResult]
    total_chunks: int
    model_name: Optional[str] = None
    processing_time_seconds: float


class HealthResponse(BaseModel):
    """Response for the /health endpoint."""

    status: Literal["healthy", "degraded"]
    active_requests: int
    waiting_requests: int
    loaded_models: list[str]


class ModelInfo(BaseModel):
    """Information about a single embedding model."""

    name: str
    dimension: int
    max_seq_length: int
    language: str
    description: Optional[str] = None


class ModelsResponse(BaseModel):
    """Response for the /models endpoint."""

    models: list[ModelInfo]
    default_model: str
