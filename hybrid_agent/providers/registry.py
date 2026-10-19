"""Provider registry and factory."""

from __future__ import annotations

from typing import Any

from .base import BaseProvider, ProviderConfig
from .config import AppConfig, NativeHybridConfig
from .errors import ConfigurationError
from .native import NativeHybridProvider
from .opencode import OpenCodeProvider


def create_provider(
    cfg: AppConfig,
    *,
    native_model: Any = None,
    native_tokenizer: Any = None,
    transport=None,
) -> BaseProvider:
    if cfg.provider in ("opencode", "opencode-go"):
        if cfg.provider == "opencode-go" and cfg.opencode.base_url == "https://opencode.ai/inference/openai/v1":
            cfg.opencode.base_url = "https://opencode.ai/inference/go/openai/v1"
        pcfg = ProviderConfig(
            provider="opencode",
            model=cfg.opencode.model,
            base_url=cfg.opencode.base_url,
            api_key_env=cfg.opencode.api_key_env,
            timeout_seconds=cfg.opencode.timeout_seconds,
            max_retries=cfg.opencode.max_retries,
        )
        pid = "opencode-go" if cfg.provider == "opencode-go" else "opencode"
        return OpenCodeProvider(pcfg, opencode_config=cfg.opencode, transport=transport, provider_id=pid)
    if cfg.provider in ("native", "native_hybrid"):
        nh: NativeHybridConfig = cfg.native_hybrid
        pcfg = ProviderConfig(
            provider="native_hybrid",
            model=nh.model or cfg.native_hybrid.model,
            base_url=None,
            api_key_env="",
            timeout_seconds=0.0,
            max_retries=0,
        )
        return NativeHybridProvider(pcfg, model=native_model, tokenizer=native_tokenizer)
    raise ConfigurationError(f"unknown provider: {cfg.provider!r}")


def provider_from_name(name: str, **kwargs) -> BaseProvider:
    from .config import load_config

    cfg = load_config(kwargs.pop("config_path", None))
    cfg.provider = name
    if "model" in kwargs and name == "opencode":
        cfg.opencode.model = kwargs.pop("model")
    if "base_url" in kwargs:
        cfg.opencode.base_url = kwargs.pop("base_url")
    return create_provider(cfg, **{k: v for k, v in kwargs.items() if k in ("native_model", "native_tokenizer", "transport")})
