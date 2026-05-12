# Document Chunking API

FastAPI service for document chunking and sentence-transformer embeddings. It is intended for RAG pipelines, semantic search, vector database ingestion, and document indexing workflows.

The API accepts files, extracts text or tabular content, creates tokenizer-aware chunks, and optionally returns normalized embedding vectors.

## Supported Inputs

| Type | Notes |
| --- | --- |
| PDF | Text extraction with `pdfplumber`. |
| DOCX | Reads paragraphs and table cells. |
| TXT | Tries UTF-8, UTF-8 BOM, Windows-1254, ISO-8859-9, Latin-1, and ISO-8859-1. |
| CSV | Detects common delimiters and handles UTF-16 BOM / NUL bytes. |
| XLSX / XLS | Reads spreadsheets into DataFrames. |
| ZIP | Extracts supported files, including password-protected AES ZIP archives. |

## Runtime Note

The current implementation requires CUDA when `/process` loads a model or tokenizer. `Dockerfile.cpu` and `docker-compose.cpu.yml` exist, but `app.config.get_device()` raises an error if CUDA is unavailable. In practice, use the GPU Docker setup for processing requests.

## Features

- Sentence-aware chunking with tokenizer-based size checks.
- Turkish and English abbreviation handling.
- Decimal number protection, such as `3.14`.
- Configurable chunk overlap.
- Optional text normalization: `none`, `lowercase`, `uppercase`.
- Row-level CSV and Excel chunking.
- Optional row metadata attachment for tabular files.
- BGE and E5 query/passage prefix support.
- Model caching through `ModelManager`.
- JSON, base64 NPY, and ZIP response formats.
- Bearer token authentication through `CHUNKING_AUTH_TOKEN`.
- File size, timeout, and concurrency limits.
- Structured JSON logging and consistent error responses.

## Quick Start

Start the GPU service:

```bash
cp .env.example .env
docker compose up -d --build
```

With the provided `.env`, the service is available at:

```text
http://localhost:8001
```

Health check:

```bash
curl http://localhost:8001/health
```

List models:

```bash
curl http://localhost:8001/models
```

Process a document:

```bash
curl -X POST http://localhost:8001/process \
  -H "Authorization: Bearer changeme-chunking-token" \
  -F "files=@document.pdf" \
  -F "model_name=BAAI/bge-m3" \
  -F "max_tokens=512" \
  -F "overlap=100"
```

Return chunks without embedding arrays:

```bash
curl -X POST http://localhost:8001/process \
  -H "Authorization: Bearer changeme-chunking-token" \
  -F "files=@document.pdf" \
  -F "skip_embedding=true"
```

## API

### `POST /process`

Processes uploaded files and returns chunks with optional embeddings.

Authentication is required only when `CHUNKING_AUTH_TOKEN` is set.

Common form parameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `files` | Required | One or more uploaded files. |
| `model_name` | First configured model | Model key from `app/config.py`. |
| `normalization` | `none` | `none`, `lowercase`, or `uppercase`. |
| `max_tokens` | `512` | Maximum tokens per chunk. |
| `overlap` | `100` | Approximate token overlap between chunks. |
| `skip_embedding` | `false` | Return chunks only. |
| `response_format` | `json` | `json`, `json_with_embeddings`, or `zip`. |
| `embedding_batch_size` | `32` | Batch size for model encoding. |
| `prefix_mode` | `passage` | `passage` or `query` for BGE/E5 style models. |
| `zip_password` | `null` | Password for encrypted ZIP inputs. |
| `output_text_column` | Empty string | Use one table column as chunk text when present. |
| `include_column_names` | `true` | Include column names in table chunk text. |
| `attach_row_data` | `true` | Add source row data to metadata. |

Response shape:

```json
{
  "results": [
    {
      "filename": "document.pdf",
      "chunks": [
        {
          "text": "First sentence. Second sentence.",
          "metadata": {
            "source_file": "document.pdf",
            "chunk_index": 0,
            "token_count": 128,
            "sentence_aware": true,
            "embedding_model_id": "BAAI/bge-m3"
          }
        }
      ],
      "embeddings": [[0.012, -0.034]]
    }
  ],
  "total_chunks": 1,
  "model_name": "BAAI/bge-m3",
  "processing_time_seconds": 1.42
}
```

### `GET /models`

Returns all configured embedding models and the default model.

### `GET /health`

Returns service status, active request count, waiting request count, and loaded model names.

## Supported Models

| Model key | Dimension | Max length | Language |
| --- | ---: | ---: | --- |
| `all-MiniLM-L6-v2` | 384 | 256 | English |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 128 | Multilingual |
| `all-mpnet-base-v2` | 768 | 384 | English |
| `bge-base-en-v1.5` | 768 | 512 | English |
| `bge-large-en-v1.5` | 1024 | 512 | English |
| `BAAI/bge-m3` | 1024 | 8192 | Multilingual |
| `multilingual-e5-large` | 1024 | 512 | Multilingual |
| `e5-base-v2` | 768 | 512 | English |
| `e5-large-v2` | 1024 | 512 | English |

For Turkish or multilingual workloads, `BAAI/bge-m3` and `multilingual-e5-large` are the most relevant starting points.

## Configuration

The service reads settings from environment variables. Docker Compose loads `.env`.

| Variable | Default | Description |
| --- | --- | --- |
| `API_PORT` | `8000` | Container application port. |
| `WORKERS` | `1` | Uvicorn worker count in Docker. |
| `LOG_LEVEL` | `INFO` | Logging level. |
| `MAX_TOKENS` | `512` | Default chunk size. |
| `OVERLAP` | `100` | Default chunk overlap. |
| `MAX_FILE_SIZE_MB` | `100` | Maximum uploaded file size. |
| `MAX_FILES` | `10` | Maximum files per request. |
| `MAX_TABLE_ROWS` | `500000` | Maximum CSV/Excel rows. |
| `MAX_CONCURRENT_REQUESTS` | `2` | Concurrent embedding limit. |
| `REQUEST_TIMEOUT_SECONDS` | `1800` | Request timeout. |
| `CONCURRENCY_ACQUIRE_TIMEOUT` | `30` | Wait time for an embedding slot. |
| `PRELOAD_MODELS` | Empty | Comma-separated model keys loaded at startup. |
| `CHUNKING_AUTH_TOKEN` | Empty | Bearer token. Empty means auth is disabled. |
| `CUDA_VISIBLE_DEVICES` | Runtime-dependent | Visible CUDA device selection. |

Example:

```env
API_PORT=8001
WORKERS=1
MAX_TOKENS=512
OVERLAP=100
MAX_FILE_SIZE_MB=100
MAX_FILES=10
MAX_CONCURRENT_REQUESTS=2
PRELOAD_MODELS=BAAI/bge-m3
CUDA_VISIBLE_DEVICES=0
CHUNKING_AUTH_TOKEN=change-this-token
```

Do not commit real Hugging Face tokens or production API tokens.

## Architecture

```text
Upload files
  -> FileProcessor
  -> TextNormalizer / TableProcessor
  -> Chunker
  -> ModelManager
  -> EmbeddingEngine
  -> ResponseFormatter
```

Main modules:

| Module | Purpose |
| --- | --- |
| `app/api/router.py` | Defines `/process`, `/models`, and `/health`. |
| `app/services/file_processor.py` | Reads supported document formats. |
| `app/services/chunker.py` | Creates sentence-aware chunks. |
| `app/services/table_processor.py` | Converts tables into row-level chunks. |
| `app/services/model_manager.py` | Loads and caches models/tokenizers. |
| `app/services/embedding.py` | Generates embeddings. |
| `app/services/response_formatter.py` | Builds JSON, base64 NPY, and ZIP responses. |
| `app/middleware/*` | Handles file size checks, timeouts, and errors. |

## Response Formats

- `json`: returns embeddings as nested numeric arrays.
- `json_with_embeddings`: returns embeddings as base64-encoded NPY data.
- `zip`: returns `results.json` plus one `.npy` file per processed input.

## Errors

Common error codes:

| Status | Code | Meaning |
| ---: | --- | --- |
| 400 | `INVALID_PARAMETER` | Request validation failed. |
| 400 | `INVALID_FILE` | File could not be parsed. |
| 400 | `ZIP_PASSWORD_REQUIRED` | ZIP requires a password. |
| 400 | `ZIP_WRONG_PASSWORD` | ZIP password is incorrect. |
| 408 | `REQUEST_TIMEOUT` | Request exceeded timeout. |
| 413 | `FILE_TOO_LARGE` | File or request is too large. |
| 422 | `UNSUPPORTED_FORMAT` | File extension is not supported. |
| 422 | `TOO_MANY_ROWS` | Table row limit exceeded. |
| 429 | `TOO_MANY_REQUESTS` | Concurrency limit is saturated. |
| 500 | `INTERNAL_ERROR` | Unexpected server error. |

## Local Development

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

For full `/process` functionality, use a CUDA-enabled environment and install a compatible PyTorch build.
