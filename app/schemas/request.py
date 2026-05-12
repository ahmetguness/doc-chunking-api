"""Request schemas for the Document Chunking & Embedding API."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ProcessParams(BaseModel):
    """Parameters for the /process endpoint.

    Default values are kept in sync with APP_CONFIG["default_values"]
    in config.py.  This schema is used for documentation / validation.
    """

    model_name: Optional[str] = Field(
        default=None, description="Embedding model adı"
    )
    normalization: Literal["none", "lowercase", "uppercase"] = Field(
        default="none"
    )
    max_tokens: int = Field(default=512, ge=1, le=32768)
    overlap: int = Field(default=100, ge=0)
    skip_embedding: bool = Field(default=False)
    response_format: Literal["json", "json_with_embeddings", "zip"] = Field(
        default="json"
    )
    embedding_batch_size: int = Field(default=32, ge=1, le=512)
    prefix_mode: Literal["passage", "query"] = Field(
        default="passage",
        description=(
            "BGE/E5 gibi instruction-prefix gerektiren modeller için "
            "hangi prefix'in kullanılacağını belirler. 'passage' doküman "
            "indexleme, 'query' arama sorguları içindir."
        ),
    )
    zip_password: Optional[str] = Field(default=None)
    include_column_names: bool = Field(default=True)
    attach_row_data: bool = Field(default=True)
    flatten_row_values_to_root: bool = Field(default=True)
    output_text_column: str = Field(
        default="",
        description=(
            "Eğer bu sütun adı DataFrame'de varsa, sadece o sütunun değeri "
            "chunk text'i olarak kullanılır. Boş bırakılırsa tüm sütunlar birleştirilir."
        ),
    )
    values_only_threshold: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Bu sayıda veya daha az boş olmayan sütun varsa sadece değerleri yaz",
    )
    attach_context_on_split: bool = Field(
        default=True,
        description=(
            "Satır chunksize'ı aştığında split edilen parçalara "
            "kısa sütun değerlerini bağlam olarak ekler."
        ),
    )
    min_cols_for_table: Optional[int] = Field(default=2, ge=1)
    min_rows_for_table: Optional[int] = Field(default=2, ge=1)
