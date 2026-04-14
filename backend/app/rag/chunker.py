"""Contract PDF chunker: 512 tokens, 50-token overlap (Blueprint §2.3).

Stage 4 implementation.
"""

from __future__ import annotations

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Split text into overlapping token-approximate chunks.

    Uses whitespace tokenisation as an approximation; Stage 4 will switch to a
    proper tokeniser (e.g. tiktoken or sentence-transformers AutoTokenizer).
    """
    words = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks
