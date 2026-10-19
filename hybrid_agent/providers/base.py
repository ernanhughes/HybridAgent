"""Base provider interface and normalized records."""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .capabilities import Capabilities, STRUCTURED_GENERATION, TEXT_GENERATION, TOOL_CALLS
from .errors import UnsupportedCapabilityError


@dataclass
class Usage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    raw: dict | None = None

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class ProviderRequest:
    messages: list[dict]
    model: str | None = None
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int | None = None
    response_format: dict | None = None
    tools: list[dict] | None = None
    reasoning_effort: str | None = None
    stream: bool = False
    metadata: dict = field(default_factory=dict)
    required_capabilities: set[str] = field(default_factory=set)

    def required(self) -> set[str]:
        req = set(self.required_capabilities)
        req.add(TEXT_GENERATION)
        if self.response_format is not None:
            req.add(STRUCTURED_GENERATION)
        if self.tools:
            req.add(TOOL_CALLS)
        return req


@dataclass
class ProviderResponse:
    text: str
    model: str
    provider: str
    usage: Usage | None = None
    raw: dict | None = None
    latency_s: float = 0.0
    request_id: str | None = None
    created: float = field(default_factory=lambda: time.time())
    metadata: dict = field(default_factory=dict)
    execution_mode: str = "TEXT_API"

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "model": self.model,
            "provider": self.provider,
            "usage": self.usage.to_dict() if self.usage else None,
            "latency_s": self.latency_s,
            "request_id": self.request_id,
            "execution_mode": self.execution_mode,
        }


@dataclass
class ProviderConfig:
    provider: str
    model: str | None = None
    base_url: str | None = None
    api_key_env: str = "OPENCODE_API_KEY"
    timeout_seconds: float = 120.0
    max_retries: int = 2
    extra: dict = field(default_factory=dict)


class BaseProvider(ABC):
    def __init__(self, config: ProviderConfig, capabilities: Capabilities):
        self.config = config
        self.capabilities = capabilities
        self._created = time.time()

    @property
    @abstractmethod
    def provider_id(self) -> str: ...

    @property
    def model_id(self) -> str:
        return self.config.model or ""

    def provenance(self) -> dict:
        return {
            "provider": self.provider_id,
            "model": self.model_id,
            "capabilities": self.capabilities.to_dict(),
            "created": self._created,
        }

    def check_capabilities(self, request: ProviderRequest) -> None:
        required = request.required()
        if not self.capabilities.supports(required):
            missing = required - self.capabilities.as_set()
            raise UnsupportedCapabilityError(
                f"Provider {self.provider_id!r} lacks capabilities: {sorted(missing)}"
            )

    @abstractmethod
    def generate(self, request: ProviderRequest) -> ProviderResponse: ...

    def check_config(self) -> dict:
        """Return a diagnostic dict without contacting the API."""
        return {
            "provider": self.provider_id,
            "model": self.model_id,
            "model_configured": bool(self.config.model),
            "capabilities": self.capabilities.to_dict(),
            "base_url": self.config.base_url,
            "api_key_env": self.config.api_key_env,
            "api_key_present": self._api_key_present(),
            "timeout_seconds": self.config.timeout_seconds,
            "max_retries": self.config.max_retries,
        }

    def _api_key_present(self) -> bool:
        import os

        return bool(os.environ.get(self.config.api_key_env))

    def close(self) -> None:
        return None

    @staticmethod
    def new_request_id() -> str:
        return str(uuid.uuid4())
