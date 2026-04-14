"""BGE-large-en-v1.5 embedding model (Blueprint §2.3, §1.5).

Stage 4 implementation.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

# BGE instruction prefix required for retrieval embeddings
_BGE_PREFIX = "Represent this document for retrieval: "
_MODEL_NAME = "BAAI/bge-large-en-v1.5"
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed(text: str) -> list[float]:
    """Embed a single query or document chunk (1024-dim)."""
    model = get_model()
    prefixed = _BGE_PREFIX + text
    return model.encode(prefixed, normalize_embeddings=True).tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts."""
    model = get_model()
    prefixed = [_BGE_PREFIX + t for t in texts]
    return model.encode(prefixed, normalize_embeddings=True).tolist()
