"""CSV/Excel row-based table processor."""

import logging
from typing import Optional

import pandas as pd

from app.schemas.internal import Chunk
from app.services.chunker import Chunker
from app.services.normalizer import TextNormalizer

logger = logging.getLogger(__name__)


class TableProcessor:
    """Processes CSV/Excel DataFrames into row-based chunks."""

    def __init__(self) -> None:
        self._chunker = Chunker()
        self._normalizer = TextNormalizer()

    def process(
        self,
        df: pd.DataFrame,
        filename: str,
        file_extension: str,
        tokenizer,
        include_column_names: bool = True,
        attach_row_data: bool = False,
        flatten_row_values_to_root: bool = False,
        output_text_column: str = "",
        values_only_threshold: int = 2,
        min_cols_for_table: Optional[int] = None,
        min_rows_for_table: Optional[int] = None,
        max_tokens: int = 512,
        overlap: int = 0,
        normalization: str = "none",
        attach_context_on_split: bool = True,
    ) -> list[Chunk]:
        if df.empty:
            return []

        if min_cols_for_table is not None and len(df.columns) < min_cols_for_table:
            return self._fallback_unstructured(
                df, filename, tokenizer, max_tokens, overlap, normalization
            )
        if min_rows_for_table is not None and len(df) < min_rows_for_table:
            return self._fallback_unstructured(
                df, filename, tokenizer, max_tokens, overlap, normalization
            )

        columns = [str(c) for c in df.columns]
        table_type = self._resolve_table_type(file_extension)

        # --- Korumalar ---
        # Tamamen boş kolonları kaldır (chunk text'te noise önlenir)
        df = df.dropna(axis=1, how="all")
        columns = [str(c) for c in df.columns]

        # str_df'i bir kez oluştur - hem filtre hem _build_all_row_texts kullanır
        str_df = df.fillna("").astype(str)

        # Tamamen boş string satırları kaldır
        str_mask = str_df.apply(lambda row: row.str.strip().any(), axis=1)
        df = df[str_mask].reset_index(drop=True)
        str_df = str_df[str_mask].reset_index(drop=True)

        if df.empty:
            logger.warning("TableProcessor: %s - tüm satırlar boş", filename)
            return []

        logger.info(
            "TableProcessor: %s - %d satır × %d kolon işlenecek",
            filename, len(df), len(columns),
        )

        # --- Vektörel metin oluşturma ---
        row_texts, str_df = self._build_all_row_texts(
            df, columns, include_column_names, output_text_column,
            values_only_threshold, normalization, str_df,
        )

        # --- Batch token tahminleme ---
        char_token_ratio = self._estimate_char_token_ratio(row_texts, tokenizer)
        max_chars_estimate = int(max_tokens * char_token_ratio * 1.1)  # %10 buffer

        logger.info(
            "TableProcessor: %d satır, char/token ratio=%.1f, max_chars_estimate=%d",
            len(row_texts), char_token_ratio, max_chars_estimate,
        )

        # --- Chunk oluşturma ---
        chunks: list[Chunk] = []
        chunk_index = 0

        # Row data hazırla (attach_row_data veya flatten için)
        need_row_data = attach_row_data or flatten_row_values_to_root
        row_dicts = None
        if need_row_data:
            row_dicts = str_df.to_dict("records")

        for i, row_text in enumerate(row_texts):
            if not row_text:
                continue

            text_len = len(row_text)

            # Hızlı yol: karakter sayısı güvenli sınırın altındaysa tokenizer çağırma
            if text_len <= max_chars_estimate:
                token_count = max(1, int(text_len / char_token_ratio))

                chunk = Chunk(
                    text=row_text,
                    source_file=filename,
                    chunk_index=chunk_index,
                    token_count=token_count,
                    is_tabular=True,
                    table_type=table_type,
                    single_row=True,
                    row_split=False,
                    table_columns=columns,
                )
                if need_row_data and row_dicts:
                    self._attach_row_metadata(
                        chunk, row_dicts[i], attach_row_data, flatten_row_values_to_root
                    )
                chunks.append(chunk)
                chunk_index += 1
            else:
                # Büyük satır - ratio ile hızlı split
                estimated_tokens = max(1, int(text_len / char_token_ratio))

                if estimated_tokens <= max_tokens:
                    chunk = Chunk(
                        text=row_text,
                        source_file=filename,
                        chunk_index=chunk_index,
                        token_count=estimated_tokens,
                        is_tabular=True,
                        table_type=table_type,
                        single_row=True,
                        row_split=False,
                        table_columns=columns,
                    )
                    if need_row_data and row_dicts:
                        self._attach_row_metadata(
                            chunk, row_dicts[i], attach_row_data, flatten_row_values_to_root
                        )
                    chunks.append(chunk)
                    chunk_index += 1
                else:
                    # Karakter bazlı hızlı split (tokenizer çağırmadan)
                    chars_per_chunk = int(max_tokens * char_token_ratio)

                    # Bağlam prefix'i hazırla (attach_context_on_split=True ise)
                    prefix = ""
                    prefix_len = 0
                    if attach_context_on_split:
                        prefix_parts = []
                        row_vals = str_df.values[i] if str_df is not None else []
                        for col, val in zip(columns, row_vals):
                            val_stripped = val.strip()
                            if val_stripped and len(val_stripped) <= chars_per_chunk // 3:
                                if include_column_names:
                                    prefix_parts.append(f"{col}: {val_stripped}")
                                else:
                                    prefix_parts.append(val_stripped)
                        if prefix_parts:
                            prefix = " | ".join(prefix_parts) + " | "
                            prefix_len = len(prefix)
                            # Prefix'i chunk boyutunun max %25'i ile sınırla
                            max_prefix_chars = chars_per_chunk // 4
                            if prefix_len > max_prefix_chars:
                                shortened = []
                                current_len = 0
                                for part in prefix_parts:
                                    if current_len + len(part) + 3 > max_prefix_chars:
                                        break
                                    shortened.append(part)
                                    current_len += len(part) + 3
                                prefix = " | ".join(shortened) + " | " if shortened else ""
                                prefix_len = len(prefix)

                    pos = 0
                    is_first = True

                    while pos < text_len:
                        if is_first:
                            end = min(pos + chars_per_chunk, text_len)
                        else:
                            effective_chars = max(chars_per_chunk - prefix_len, chars_per_chunk // 2)
                            end = min(pos + effective_chars, text_len)

                        # Kelime sınırına snap
                        if end < text_len:
                            space_pos = row_text.rfind(" ", pos, end)
                            if space_pos > pos:
                                end = space_pos + 1

                        segment = row_text[pos:end].strip()
                        if segment:
                            if not is_first and prefix:
                                segment = prefix + segment

                            seg_tokens = max(1, int(len(segment) / char_token_ratio))
                            chunk = Chunk(
                                text=segment,
                                source_file=filename,
                                chunk_index=chunk_index,
                                token_count=seg_tokens,
                                is_tabular=True,
                                table_type=table_type,
                                single_row=is_first and end >= text_len,
                                row_split=True,
                                table_columns=columns,
                            )
                            if need_row_data and row_dicts:
                                self._attach_row_metadata(
                                    chunk, row_dicts[i], attach_row_data, flatten_row_values_to_root
                                )
                            chunks.append(chunk)
                            chunk_index += 1
                            is_first = False
                        pos = end

        return chunks

    # ------------------------------------------------------------------
    # Vektörel metin oluşturma
    # ------------------------------------------------------------------

    def _build_all_row_texts(
        self,
        df: pd.DataFrame,
        columns: list[str],
        include_column_names: bool,
        output_text_column: str,
        values_only_threshold: int,
        normalization: str,
        str_df: Optional[pd.DataFrame] = None,
    ) -> tuple[list[str], pd.DataFrame]:
        """Tüm satırları metin haline getir. (str_df de döner, çift kopya önlenir)"""

        if str_df is None:
            str_df = df.fillna("").astype(str)

        # output_text_column varsa direkt o sütunu al
        if output_text_column and output_text_column in df.columns:
            texts = str_df[output_text_column].tolist()
            if normalization == "lowercase":
                texts = [t.lower().strip() for t in texts]
            elif normalization == "uppercase":
                texts = [t.upper().strip() for t in texts]
            else:
                texts = [t.strip() for t in texts]
            return texts, str_df

        # numpy array - iloc yerine direkt erişim
        values_array = str_df.values

        texts = []
        for idx in range(len(values_array)):
            row_values = values_array[idx]
            non_empty = [(col, val) for col, val in zip(columns, row_values) if val.strip()]

            if not non_empty:
                texts.append("")
                continue

            if len(non_empty) <= values_only_threshold:
                text = " | ".join(val for _, val in non_empty)
            elif include_column_names:
                text = " | ".join(f"{col}: {val}" for col, val in non_empty)
            else:
                text = " | ".join(val for _, val in non_empty)

            if normalization == "lowercase":
                text = text.lower().strip()
            elif normalization == "uppercase":
                text = text.upper().strip()
            else:
                text = text.strip()

            texts.append(text)

        return texts, str_df

    # ------------------------------------------------------------------
    # Token ratio tahminleme
    # ------------------------------------------------------------------

    def _estimate_char_token_ratio(
        self, texts: list[str], tokenizer, sample_size: int = 100
    ) -> float:
        """Sample'dan karakter/token oranını hesapla."""
        non_empty = [t for t in texts if t]
        if not non_empty:
            return 4.0  # fallback

        step = max(1, len(non_empty) // sample_size)
        samples = non_empty[::step][:sample_size]

        total_chars = 0
        total_tokens = 0
        for text in samples:
            total_chars += len(text)
            try:
                tokens = tokenizer.encode(text, add_special_tokens=False)
                total_tokens += len(tokens)
            except Exception:
                total_tokens += max(1, len(text.split()))

        if total_tokens == 0:
            return 4.0

        return total_chars / total_tokens

    # ------------------------------------------------------------------
    # Unstructured fallback
    # ------------------------------------------------------------------

    def _fallback_unstructured(
        self,
        df: pd.DataFrame,
        filename: str,
        tokenizer,
        max_tokens: int,
        overlap: int,
        normalization: str,
    ) -> list[Chunk]:
        columns = [str(c) for c in df.columns]
        header = " | ".join(columns)
        str_df = df.fillna("").astype(str)
        lines = [header]
        for idx in range(len(str_df)):
            values = str_df.iloc[idx].tolist()
            lines.append(" | ".join(values))
        text = "\n".join(lines)
        text = self._normalizer.normalize(text, mode=normalization)
        if not text:
            return []
        return self._chunker.chunk(
            text=text,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            overlap=overlap,
            filename=filename,
        )

    # ------------------------------------------------------------------
    # Metadata + utility
    # ------------------------------------------------------------------

    @staticmethod
    def _attach_row_metadata(
        chunk: Chunk,
        row_dict: dict,
        attach_row_data: bool,
        flatten_row_values_to_root: bool,
    ) -> None:
        if attach_row_data:
            chunk.row_data = {str(k): str(v) for k, v in row_dict.items()}
        if flatten_row_values_to_root:
            chunk.extra_metadata = {str(k): str(v) for k, v in row_dict.items()}

    @staticmethod
    def _resolve_table_type(file_extension: str) -> str:
        mapping = {".csv": "csv", ".xlsx": "excel", ".xls": "excel"}
        return mapping.get(file_extension.lower(), "table")

