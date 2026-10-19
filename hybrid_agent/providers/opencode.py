"""OpenCode hosted inference provider (OpenAI Chat Completions compatible)."""

from __future__ import annotations

import os
import time
from typing import Any

from .base import BaseProvider, ProviderConfig, ProviderRequest, ProviderResponse, Usage
from .capabilities import Capabilities
from .config import OpenCodeConfig
from .errors import (
    AuthenticationError,
    MalformedResponseError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
    TimeoutError as ProviderTimeoutError,
)
from .transport import HttpResponse, Transport, default_transport


class OpenCodeProvider(BaseProvider):
    def __init__(
        self,
        config: ProviderConfig,
        opencode_config: OpenCodeConfig | None = None,
        transport: Transport | None = None,
        provider_id: str = "opencode",
        session_id: str | None = None,
    ):
        self.opencode_config = opencode_config or OpenCodeConfig(
            base_url=config.base_url or "https://opencode.ai/inference/openai/v1",
            model=config.model,
            api_key_env=config.api_key_env,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
        )
        if config.base_url:
            self.opencode_config.base_url = config.base_url
        if config.model is not None:
            self.opencode_config.model = config.model
        self.opencode_config.api_key_env = config.api_key_env
        self.opencode_config.timeout_seconds = config.timeout_seconds
        self.opencode_config.max_retries = config.max_retries

        self._provider_id = provider_id
        import os as _os
        import uuid as _uuid
        self.session_id = session_id or _os.environ.get("OPENCODE_SESSION_ID") or str(_uuid.uuid4())
        caps = Capabilities(
            text_generation=True,
            structured_generation=self.opencode_config.supports_structured_output,
            tool_calls=self.opencode_config.supports_tool_calls,
            usage_reporting=True,
            latent_reasoning=False,
            hidden_state_access=False,
            latent_readout=False,
            latent_exit_trace=False,
            hybridlink_access=False,
        )
        super().__init__(config, caps)
        self._transport = transport or default_transport

    @property
    def provider_id(self) -> str:
        return self._provider_id

    def _api_key(self) -> str | None:
        return os.environ.get(self.config.api_key_env)

    def _chat_url(self) -> str:
        base = (self.opencode_config.base_url or self.config.base_url or "").rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return base + "/chat/completions"

    def _headers(self, request_id: str) -> dict:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "curl/8.0",
            "X-Request-Id": request_id,
        }
        key = self._api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        headers["x-opencode-session"] = self.session_id
        return headers

    def _body(self, request: ProviderRequest) -> dict:
        raw_model = request.model or self.opencode_config.model or self.config.model
        if not raw_model:
            raise ValueError("no model configured")
        model = str(raw_model).split("/")[-1]
        body: dict[str, Any] = {
            "model": model,
            "messages": request.messages,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens
        if request.response_format is not None:
            body["response_format"] = request.response_format
        if request.tools:
            body["tools"] = request.tools
        if request.stream:
            body["stream"] = True
        if request.reasoning_effort:
            body["reasoning_effort"] = request.reasoning_effort
        return body

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        self.check_capabilities(request)
        if not (self.opencode_config.model or self.config.model or request.model):
            from .errors import ConfigurationError

            raise ConfigurationError("no model configured for OpenCode provider")
        if not self._api_key():
            raise AuthenticationError(
                f"missing API key in environment variable {self.config.api_key_env!r}"
            )

        request_id = self.new_request_id()
        url = self._chat_url()
        body = self._body(request)
        headers = self._headers(request_id)

        max_retries = self.opencode_config.max_retries
        last_error: Exception | None = None
        t0 = time.time()
        for attempt in range(max_retries + 1):
            try:
                resp: HttpResponse = self._transport(
                    "POST",
                    url,
                    headers=headers,
                    json=body,
                    timeout=self.opencode_config.timeout_seconds,
                )
            except Exception as exc:  # transport-level
                last_error = exc
                if attempt < max_retries:
                    time.sleep(min(2 ** attempt, 4))
                    continue
                raise ProviderTimeoutError(f"transport error: {exc}") from exc

            status = resp.status_code
            if status == 401 or status == 403:
                raise AuthenticationError(f"authentication failed (HTTP {status})")
            if status == 404:
                raise ModelNotFoundError(f"model or endpoint not found (HTTP {status}): {resp.text[:200]}")
            if status == 429:
                if attempt < max_retries:
                    retry_after = resp.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after else min(2 ** attempt, 4)
                    time.sleep(delay)
                    last_error = RateLimitError("rate limited (HTTP 429)")
                    continue
                raise RateLimitError(f"rate limited (HTTP 429): {resp.text[:200]}")
            if 500 <= status < 600:
                if attempt < max_retries:
                    time.sleep(min(2 ** attempt, 4))
                    last_error = ProviderError(f"server error (HTTP {status})")
                    continue
                raise ProviderError(f"server error (HTTP {status}): {resp.text[:200]}")
            if status < 200 or status >= 300:
                raise ProviderError(f"unexpected HTTP {status}: {resp.text[:300]}")

            data = resp.data
            if not isinstance(data, dict):
                raise MalformedResponseError("response is not a JSON object")
            return self._normalize(data, request, request_id, time.time() - t0, body)

        raise (last_error or ProviderError("request failed"))

    def _normalize(
        self,
        data: dict,
        request: ProviderRequest,
        request_id: str,
        latency: float,
        body: dict,
    ) -> ProviderResponse:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise MalformedResponseError("response has no choices")
        first = choices[0]
        if not isinstance(first, dict):
            raise MalformedResponseError("choice is not an object")
        message = first.get("message")
        if not isinstance(message, dict):
            raise MalformedResponseError("choice has no message")
        text = message.get("content")
        if text is None:
            text = ""
        if not isinstance(text, str):
            text = str(text)

        usage_raw = data.get("usage")
        usage = None
        if isinstance(usage_raw, dict):
            usage = Usage(
                prompt_tokens=usage_raw.get("prompt_tokens"),
                completion_tokens=usage_raw.get("completion_tokens"),
                total_tokens=usage_raw.get("total_tokens"),
                raw=usage_raw,
            )

        model = data.get("model") or body.get("model") or self.model_id

        return ProviderResponse(
            text=text,
            model=model,
            provider=self.provider_id,
            usage=usage,
            raw=data,
            latency_s=round(latency, 4),
            request_id=request_id,
            metadata={
                "finish_reason": first.get("finish_reason"),
                "response_id": data.get("id"),
                "system_fingerprint": data.get("system_fingerprint"),
            },
            execution_mode="TEXT_API",
        )

    def check_config(self) -> dict:
        info = super().check_config()
        info["chat_url"] = self._chat_url()
        info["api_family"] = self.opencode_config.api_family
        return info
