"""Configuration loading and validation."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .errors import ConfigurationError


@dataclass
class OpenCodeConfig:
    api_family: str = "openai-chat"
    base_url: str = "https://opencode.ai/inference/openai/v1"
    model: str | None = None
    api_key_env: str = "OPENCODE_API_KEY"
    timeout_seconds: float = 120.0
    max_retries: int = 2
    supports_structured_output: bool = False
    supports_tool_calls: bool = False


@dataclass
class NativeHybridConfig:
    enabled: bool = False
    model: str | None = None


@dataclass
class ExecutionConfig:
    preserve_raw: bool = True
    dry_run: bool = False
    max_requests: int = 100
    max_cost_usd: float | None = 2.00


@dataclass
class AppConfig:
    provider: str = "native"
    opencode: OpenCodeConfig = field(default_factory=OpenCodeConfig)
    native_hybrid: NativeHybridConfig = field(default_factory=NativeHybridConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    def validate(self) -> None:
        if self.provider not in ("opencode", "opencode-go", "native", "native_hybrid"):
            raise ConfigurationError(f"unknown provider: {self.provider!r}")
        if self.opencode.timeout_seconds <= 0:
            raise ConfigurationError("opencode.timeout_seconds must be positive")
        if self.opencode.max_retries < 0:
            raise ConfigurationError("opencode.max_retries must be >= 0")
        if self.execution.max_requests < 1:
            raise ConfigurationError("execution.max_requests must be >= 1")
        if self.execution.max_cost_usd is not None and self.execution.max_cost_usd < 0:
            raise ConfigurationError("execution.max_cost_usd must be >= 0 or null")
        if self.provider in ("opencode",) and not self.opencode.model:
            # Not a hard error: diagnostic surfaces missing model. Execution will fail.
            pass


def _load_file(path: Path) -> dict:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix in (".toml",):
        try:
            import tomllib
        except ImportError:  # pragma: no cover
            import tomli as tomllib  # type: ignore
        return tomllib.loads(text)
    if suffix in (".json",):
        return json.loads(text)
    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ConfigurationError("PyYAML is required to read YAML config files") from exc
        return yaml.safe_load(text) or {}
    raise ConfigurationError(f"unsupported config format: {suffix}")


def _from_dict(data: dict) -> AppConfig:
    data = data or {}
    oc = data.get("opencode", {}) or {}
    nh = data.get("native_hybrid", {}) or {}
    ex = data.get("execution", {}) or {}
    return AppConfig(
        provider=data.get("provider", "native"),
        opencode=OpenCodeConfig(**{k: v for k, v in oc.items() if k in OpenCodeConfig.__dataclass_fields__}),
        native_hybrid=NativeHybridConfig(**{k: v for k, v in nh.items() if k in NativeHybridConfig.__dataclass_fields__}),
        execution=ExecutionConfig(**{k: v for k, v in ex.items() if k in ExecutionConfig.__dataclass_fields__}),
    )


def _apply_env(cfg: AppConfig) -> AppConfig:
    if os.environ.get("HYBRID_PROVIDER"):
        cfg.provider = os.environ["HYBRID_PROVIDER"]
    if os.environ.get("OPENCODE_BASE_URL"):
        cfg.opencode.base_url = os.environ["OPENCODE_BASE_URL"]
    if os.environ.get("OPENCODE_MODEL"):
        cfg.opencode.model = os.environ["OPENCODE_MODEL"]
    if os.environ.get("OPENCODE_API_KEY_ENV"):
        cfg.opencode.api_key_env = os.environ["OPENCODE_API_KEY_ENV"]
    if os.environ.get("OPENCODE_TIMEOUT"):
        cfg.opencode.timeout_seconds = float(os.environ["OPENCODE_TIMEOUT"])
    if os.environ.get("OPENCODE_MAX_RETRIES"):
        cfg.opencode.max_retries = int(os.environ["OPENCODE_MAX_RETRIES"])
    if os.environ.get("OPENCODE_API_FAMILY"):
        cfg.opencode.api_family = os.environ["OPENCODE_API_FAMILY"]
    return cfg


def load_config(path: str | os.PathLike | None = None) -> AppConfig:
    if path is None:
        # Search common locations.
        candidates = [
            Path("opencode.toml"),
            Path("config.toml"),
            Path("hybrid_agent.toml"),
            Path("configs/opencode.toml"),
            Path("configs/opencode.yaml"),
            Path("configs/opencode.yml"),
            Path("configs/opencode.json"),
        ]
        for c in candidates:
            if c.is_file():
                path = c
                break
    if path is None:
        cfg = AppConfig()
    else:
        cfg = _from_dict(_load_file(Path(path)))
    cfg = _apply_env(cfg)
    cfg.validate()
    return cfg


def config_to_dict(cfg: AppConfig) -> dict:
    return {
        "provider": cfg.provider,
        "opencode": asdict(cfg.opencode),
        "native_hybrid": asdict(cfg.native_hybrid),
        "execution": asdict(cfg.execution),
    }
