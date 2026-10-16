"""LLMLingua prompt compressor adapter (Paper Reference [43])."""
from typing import Optional


class LLMLinguaCompressor:
    """Compressor adapter leveraging LLMLingua / LLMLingua-2 for iterative token compression."""

    def __init__(
        self,
        model_name: str = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        device: str = "cpu",
        use_llmlingua2: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.use_llmlingua2 = use_llmlingua2
        self._compressor = None

    def _get_compressor(self):
        if self._compressor is None:
            try:
                from llmlingua import PromptCompressor
                self._compressor = PromptCompressor(
                    model_name=self.model_name,
                    device_map=self.device,
                    use_llmlingua2=self.use_llmlingua2,
                )
            except Exception:
                try:
                    from llmlingua import PromptCompressor
                    self._compressor = PromptCompressor(device_map=self.device)
                except Exception:
                    self._compressor = None
        return self._compressor

    def compress(self, text: str, rate: float = 0.5) -> str:
        if not text or not text.strip():
            return text

        compressor = self._get_compressor()
        if compressor is None:
            # Fallback heuristic if LLMLingua is not installed in lightweight environments:
            words = text.split()
            if len(words) <= 4:
                return text
            keep_count = max(2, int(len(words) * rate))
            return " ".join(words[:keep_count])

        try:
            res = compressor.compress_prompt(text, rate=rate)
            return res.get("compressed_prompt", text).strip()
        except Exception:
            return text
