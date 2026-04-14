"""Unit tests for backend.app.rag.chunker (Blueprint §2.3).

No external dependencies — pure Python, runs offline.
"""

from __future__ import annotations

from app.rag.chunker import CHUNK_OVERLAP, CHUNK_SIZE, chunk_text


class TestChunkText:
    def test_short_text_single_chunk(self) -> None:
        text = " ".join(["word"] * 10)
        chunks = chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_up_to_step_boundary_is_one_chunk(self) -> None:
        text = " ".join(["w"] * (CHUNK_SIZE - CHUNK_OVERLAP))
        chunks = chunk_text(text)
        assert len(chunks) == 1

    def test_long_text_multiple_chunks(self) -> None:
        text = " ".join(["word"] * (CHUNK_SIZE * 3))
        chunks = chunk_text(text)
        assert len(chunks) > 1

    def test_overlap_is_honoured(self) -> None:
        text = " ".join([f"w{i}" for i in range(CHUNK_SIZE + CHUNK_OVERLAP + 10)])
        chunks = chunk_text(text)
        assert len(chunks) >= 2
        # The tail of chunk 0 should match the head of chunk 1
        tail_words = chunks[0].split()[-CHUNK_OVERLAP:]
        head_words = chunks[1].split()[:CHUNK_OVERLAP]
        assert tail_words == head_words

    def test_custom_chunk_size(self) -> None:
        text = " ".join(["w"] * 100)
        chunks = chunk_text(text, chunk_size=20, overlap=5)
        # Expected: ceil((100 - 5) / (20 - 5)) ≈ 7 chunks
        assert len(chunks) >= 5
        for chunk in chunks:
            assert len(chunk.split()) <= 20

    def test_empty_string_returns_empty(self) -> None:
        assert chunk_text("") == []

    def test_single_word(self) -> None:
        chunks = chunk_text("hello")
        assert chunks == ["hello"]

    def test_default_constants(self) -> None:
        assert CHUNK_SIZE == 512
        assert CHUNK_OVERLAP == 50

    def test_no_data_loss(self) -> None:
        """Every word in the original text must appear in at least one chunk."""
        words = [f"token{i}" for i in range(600)]
        text = " ".join(words)
        chunks = chunk_text(text)
        chunked_words: set[str] = set()
        for chunk in chunks:
            chunked_words.update(chunk.split())
        assert set(words) == chunked_words
