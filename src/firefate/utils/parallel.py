"""Work-splitting helpers for the multiprocessing paths."""
from __future__ import annotations

import pandas as pd


def create_balanced_chunks(df: pd.DataFrame, n_chunks: int):
    """
    Create balanced chunks by splitting DataFrame into roughly equal parts
    """
    chunk_size = len(df) // n_chunks
    remainder = len(df) % n_chunks

    chunks = []
    start_idx = 0

    for i in range(n_chunks):
        # Add one extra row to first 'remainder' chunks
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_chunk_size

        chunk = df.iloc[start_idx:end_idx]
        if len(chunk) > 0:  # Only add non-empty chunks
            chunks.append(chunk)

        start_idx = end_idx

    return chunks
