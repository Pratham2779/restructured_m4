"""OpenAI-compatible LLM client adapter (works with Ollama, vLLM, llama.cpp)."""
from typing import List, Dict, Optional
import httpx


class OpenAICompatLLMClient:
    """
    Adapter communicating with any standard OpenAI-compatible API endpoint
    (vLLM, llama.cpp, Ollama OpenAI endpoints, etc.).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "qwen2.5:7b-instruct",
        api_key: str = "dummy",
        timeout: Optional[float] = None,
        temperature: float = 0.8,
        **kwargs,
    ):
        import os
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = float(os.environ.get("M4_LLM_TIMEOUT", 300.0)) if timeout is None else timeout
        self.temperature = temperature
        self._client = httpx.Client(timeout=self.timeout)

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        json_mode: bool = False,
        max_tokens: Optional[int] = None,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        resp = self._client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


OpenAICompatLLM = OpenAICompatLLMClient

