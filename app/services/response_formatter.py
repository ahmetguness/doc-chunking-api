"""Response formatter service for building API responses in different formats.

Supports three output formats:
- JSON: embeddings as list[list[float]] in each FileResult
- JSON with embeddings: base64-encoded NPY format in embeddings_base64_npy field
- ZIP: archive containing JSON data + per-file .npy embedding files
"""

import base64
import io
import logging
import zipfile
from typing import Optional

import numpy as np
from starlette.responses import StreamingResponse

from app.schemas.internal import Chunk
from app.schemas.response import (
    ChunkMetadata,
    ChunkResult,
    FileResult,
    ProcessResponse,
)

logger = logging.getLogger(__name__)


class ResponseFormatter:
    """Formats processing results into the requested response format."""

    def _build_chunk_results(self, chunks: list[Chunk]) -> list[ChunkResult]:
        """Convert internal Chunk objects to ChunkResult response models."""
        results = []
        for chunk in chunks:
            metadata = ChunkMetadata(
                source_file=chunk.source_file,
                chunk_index=chunk.chunk_index,
                token_count=chunk.token_count,
                is_tabular=chunk.is_tabular,
                table_type=chunk.table_type,
                single_row=chunk.single_row,
                row_split=chunk.row_split,
                sentence_aware=chunk.sentence_aware,
                table_columns=chunk.table_columns,
                embedding_model_id=chunk.embedding_model_id,
                row_data=chunk.row_data,
                extra_metadata=chunk.extra_metadata,
            )
            results.append(ChunkResult(text=chunk.text, metadata=metadata))
        return results

    def _build_file_result(
        self,
        filename: str,
        chunks: list[Chunk],
        embeddings: Optional[np.ndarray] = None,
    ) -> FileResult:
        """Build a FileResult with embeddings as list[list[float]]."""
        chunk_results = self._build_chunk_results(chunks)
        emb_list: Optional[list[list[float]]] = None
        if embeddings is not None and embeddings.size > 0:
            emb_list = embeddings.tolist()
        return FileResult(
            filename=filename,
            chunks=chunk_results,
            embeddings=emb_list,
        )

    def _build_file_result_with_base64(
        self,
        filename: str,
        chunks: list[Chunk],
        embeddings: Optional[np.ndarray] = None,
    ) -> FileResult:
        """Build a FileResult with embeddings as base64-encoded NPY string."""
        chunk_results = self._build_chunk_results(chunks)
        b64_npy: Optional[str] = None
        if embeddings is not None and embeddings.size > 0:
            buf = io.BytesIO()
            np.save(buf, embeddings)
            b64_npy = base64.b64encode(buf.getvalue()).decode("ascii")
        return FileResult(
            filename=filename,
            chunks=chunk_results,
            embeddings_base64_npy=b64_npy,
        )

    def format_json(
        self,
        file_results: list[dict],
        model_name: Optional[str] = None,
        processing_time: float = 0.0,
    ) -> ProcessResponse:
        """Format results as pure JSON with embeddings as list[list[float]].

        Args:
            file_results: List of dicts with keys:
                - filename (str)
                - chunks (list[Chunk])
                - embeddings (Optional[np.ndarray])
                - error (Optional[ErrorResponse])
            model_name: Name of the embedding model used.
            processing_time: Total processing time in seconds.

        Returns:
            ProcessResponse with embeddings in list[list[float]] format.
        """
        results: list[FileResult] = []
        total_chunks = 0

        for fr in file_results:
            if fr.get("error"):
                results.append(
                    FileResult(
                        filename=fr["filename"],
                        chunks=[],
                        error=fr["error"],
                    )
                )
                continue

            file_result = self._build_file_result(
                filename=fr["filename"],
                chunks=fr["chunks"],
                embeddings=fr.get("embeddings"),
            )
            results.append(file_result)
            total_chunks += len(file_result.chunks)

        return ProcessResponse(
            results=results,
            total_chunks=total_chunks,
            model_name=model_name,
            processing_time_seconds=processing_time,
        )

    def format_json_with_embeddings(
        self,
        file_results: list[dict],
        model_name: Optional[str] = None,
        processing_time: float = 0.0,
    ) -> ProcessResponse:
        """Format results as JSON with base64-encoded NPY embeddings.

        Args:
            file_results: Same structure as format_json.
            model_name: Name of the embedding model used.
            processing_time: Total processing time in seconds.

        Returns:
            ProcessResponse with embeddings_base64_npy field populated.
        """
        results: list[FileResult] = []
        total_chunks = 0

        for fr in file_results:
            if fr.get("error"):
                results.append(
                    FileResult(
                        filename=fr["filename"],
                        chunks=[],
                        error=fr["error"],
                    )
                )
                continue

            file_result = self._build_file_result_with_base64(
                filename=fr["filename"],
                chunks=fr["chunks"],
                embeddings=fr.get("embeddings"),
            )
            results.append(file_result)
            total_chunks += len(file_result.chunks)

        return ProcessResponse(
            results=results,
            total_chunks=total_chunks,
            model_name=model_name,
            processing_time_seconds=processing_time,
        )

    def format_zip(
        self,
        file_results: list[dict],
        model_name: Optional[str] = None,
        processing_time: float = 0.0,
    ) -> StreamingResponse:
        """Format results as a ZIP archive containing JSON + NPY files.

        The ZIP archive contains:
        - results.json: ProcessResponse JSON (without raw embeddings)
        - {filename}.npy: Per-file numpy embedding arrays

        Args:
            file_results: Same structure as format_json.
            model_name: Name of the embedding model used.
            processing_time: Total processing time in seconds.

        Returns:
            StreamingResponse with media_type="application/zip".
        """
        results: list[FileResult] = []
        total_chunks = 0
        npy_files: list[tuple[str, bytes]] = []

        for fr in file_results:
            if fr.get("error"):
                results.append(
                    FileResult(
                        filename=fr["filename"],
                        chunks=[],
                        error=fr["error"],
                    )
                )
                continue

            chunk_results = self._build_chunk_results(fr["chunks"])
            embeddings: Optional[np.ndarray] = fr.get("embeddings")

            file_result = FileResult(
                filename=fr["filename"],
                chunks=chunk_results,
            )
            results.append(file_result)
            total_chunks += len(chunk_results)

            if embeddings is not None and embeddings.size > 0:
                npy_buf = io.BytesIO()
                np.save(npy_buf, embeddings)
                npy_files.append((fr["filename"], npy_buf.getvalue()))

        response_data = ProcessResponse(
            results=results,
            total_chunks=total_chunks,
            model_name=model_name,
            processing_time_seconds=processing_time,
        )

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("results.json", response_data.model_dump_json(indent=2))
            for filename, npy_data in npy_files:
                base_name = filename.rsplit(".", 1)[0] if "." in filename else filename
                zf.writestr(f"{base_name}.npy", npy_data)

        zip_buf.seek(0)

        return StreamingResponse(
            zip_buf,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=results.zip"},
        )
