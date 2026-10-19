"""Native HybridAgent provider compatibility layer."""

from __future__ import annotations

import time
from typing import Any

from .base import BaseProvider, ProviderConfig, ProviderRequest, ProviderResponse
from .capabilities import Capabilities
from .errors import ProviderError


class NativeHybridProvider(BaseProvider):
    def __init__(
        self,
        config: ProviderConfig,
        model: Any = None,
        tokenizer: Any = None,
    ):
        caps = Capabilities(
            text_generation=model is not None,
            structured_generation=False,
            tool_calls=False,
            usage_reporting=False,
            latent_reasoning=True,
            hidden_state_access=True,
            latent_readout=True,
            latent_exit_trace=True,
            hybridlink_access=True,
        )
        super().__init__(config, caps)
        self.model = model
        self.tokenizer = tokenizer

    @property
    def provider_id(self) -> str:
        return "native_hybrid"

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        self.check_capabilities(request)
        if self.model is None or self.tokenizer is None:
            raise ProviderError("native hybrid model is not loaded")
        # Lazy import to keep module import lightweight.
        import torch

        messages = request.messages
        prompt = ""
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            prompt += f"{role}: {content}\n"
        tokenizer = self.tokenizer
        model = self.model
        device = getattr(model, "device", None)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens or 256,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
            )
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return ProviderResponse(
            text=text,
            model=self.config.model or "native",
            provider=self.provider_id,
            usage=None,
            raw=None,
            latency_s=round(time.time() - t0, 4),
            request_id=self.new_request_id(),
            execution_mode="NATIVE_HYBRID",
        )

    def check_config(self) -> dict:
        info = super().check_config()
        info["model_loaded"] = self.model is not None
        return info
