"""SentenceTransformers offline local embedder adapter."""
from typing import List, Optional


class SentenceTransformersEmbedder:
    """Local dense embedding adapter using sentence-transformers (e.g. bge-small-en-v1.5)."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "cpu",
        dim: int = 384,
    ):
        self.model_name = model_name
        self.device = device
        self.dim = dim
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self.dim = self._model.get_sentence_embedding_dimension()
        return self._model

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        model = self._get_model()
        vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return [v.tolist() for v in vectors]
