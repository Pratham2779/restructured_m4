"""Native Ollama LLM client adapter."""
from typing import List, Dict, Optional, Any
import httpx


class OllamaLLMClient:
    """
    Direct Ollama API adapter querying /api/chat with fallback to /v1/chat/completions.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:7b-instruct",
        timeout: float = 180.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        json_mode: bool = False,
        max_tokens: Optional[int] = None,
    ) -> str:
        url = f"{self.base_url}/api/chat"
        options: Dict[str, Any] = {
            "temperature": 0.8 if temperature is None else temperature,
        }
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": options,
            "keep_alive": "24h",
        }
        if json_mode:
            payload["format"] = "json"

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json=payload)
            if resp.status_code == 404:
                # Fallback to OpenAI-compatible endpoint
                url_v1 = f"{self.base_url}/v1/chat/completions"
                payload_v1 = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": options["temperature"],
                }
                if max_tokens is not None:
                    payload_v1["max_tokens"] = max_tokens
                resp2 = client.post(url_v1, json=payload_v1)
                resp2.raise_for_status()
                return resp2.json()["choices"][0]["message"]["content"]

            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"]
