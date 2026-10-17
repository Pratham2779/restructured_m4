"""SentenceTransformers offline local embedder adapter."""
from typing import List, Optional
import threading


class SentenceTransformersEmbedder:
    """Local dense embedding adapter using sentence-transformers (e.g. SeanLee97/mxbai-embed-large-v1-nli-matryoshka or BAAI/bge-small-en-v1.5)."""

    def __init__(
        self,
        model_name: str = "SeanLee97/mxbai-embed-large-v1-nli-matryoshka",
        device: Optional[str] = None,
        dim: Optional[int] = None,
        truncate_dim: Optional[int] = None,
    ):
        self.model_name = model_name
        if device is None:
            try:
                import torch
                if torch.cuda.device_count() > 1:
                    self.device = "cuda:1"
                elif torch.cuda.is_available():
                    self.device = "cuda:0"
                else:
                    self.device = "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        self._dim = dim or truncate_dim
        self.truncate_dim = truncate_dim
        self._model = None
        self._lock = threading.Lock()

    def _get_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer
                    init_kwargs = {"device": self.device}
                    if self.truncate_dim:
                        init_kwargs["truncate_dim"] = self.truncate_dim
                    self._model = SentenceTransformer(self.model_name, **init_kwargs)
                    if self._dim is None:
                        self._dim = self.truncate_dim or self._model.get_sentence_embedding_dimension()
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._get_model()
        return self._dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        model = self._get_model()
        with self._lock:
            encode_kwargs = {"convert_to_numpy": True, "normalize_embeddings": True}
            if self.truncate_dim:
                encode_kwargs["truncate_dim"] = self.truncate_dim
            vectors = model.encode(texts, **encode_kwargs)
        return [v.tolist() for v in vectors]

