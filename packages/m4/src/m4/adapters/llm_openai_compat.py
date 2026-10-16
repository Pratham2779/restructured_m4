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
        max_retries: int = 5,
    ) -> str:
        import time

        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        last_exc = None
        for attempt in range(max_retries):
            # On retries, drop response_format constraint if Ollama/llama.cpp grammar parser fails
            use_json = json_mode if attempt == 0 else False

            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature if temperature is None else temperature,
                "max_tokens": max_tokens if max_tokens is not None else 2048,
            }
            if use_json:
                payload["response_format"] = {"type": "json_object"}

            try:
                resp = self._client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                else:
                    raise last_exc
        raise last_exc


OpenAICompatLLM = OpenAICompatLLMClient

