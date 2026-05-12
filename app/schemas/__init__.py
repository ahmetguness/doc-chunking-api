"""Schema exports for the Document Chunking & Embedding API."""

from app.schemas.internal import Chunk, FileContent
from app.schemas.request import ProcessParams
from app.schemas.response import (
    ChunkMetadata,
    ChunkResult,
    ErrorResponse,
    FileResult,
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    ProcessResponse,
)

__all__ = [
    # Request
    "ProcessParams",
    # Response
    "ChunkMetadata",
    "ChunkResult",
    "ErrorResponse",
    "FileResult",
    "HealthResponse",
    "ModelInfo",
    "ModelsResponse",
    "ProcessResponse",
    # Internal
    "Chunk",
    "FileContent",
]
