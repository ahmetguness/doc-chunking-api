"""Internal data structures used across services."""

from dataclasses import dataclass, field
from typing import Literal, Optional

import pandas as pd


@dataclass
class Chunk:
    """Internal representation of a text chunk."""

    text: str
    source_file: str
    chunk_index: int
    token_count: int
    is_tabular: bool = False
    table_type: Optional[str] = None
    single_row: bool = False
    row_split: bool = False
    sentence_aware: bool = False
    table_columns: Optional[list[str]] = None
    embedding_model_id: Optional[str] = None
    row_data: Optional[dict] = None
    extra_metadata: Optional[dict] = None


@dataclass
class FileContent:
    """Represents a file's extracted content."""

    filename: str
    content: str
    file_type: Literal["text", "table"]
    file_extension: str  # Original file extension (.csv, .xlsx, .pdf, .docx, .txt)
    dataframe: Optional[pd.DataFrame] = field(default=None, repr=False)
