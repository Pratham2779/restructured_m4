"""Ollama dense text embedder adapter."""
from typing import List
import httpx


class OllamaEmbedder:
    """Computes text embeddings via Ollama /api/embed endpoint."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "bge-small-en-v1.5",
        dim: int = 384,
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dim = dim
        self.timeout = timeout
        self._client = httpx.Client(timeout=self.timeout)

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        base = self.base_url.rstrip("/")
        for suffix in ("/api/embed", "/api/embeddings", "/v1"):
            if base.endswith(suffix):
                base = base[:-len(suffix)]
        url = f"{base}/api/embed"
        payload = {
            "model": self.model,
            "input": texts,
            "keep_alive": "24h",
        }

        resp = self._client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings", [])
        return embeddings
