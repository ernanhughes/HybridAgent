"""Capability definitions and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

TEXT_GENERATION = "text_generation"
STRUCTURED_GENERATION = "structured_generation"
TOOL_CALLS = "tool_calls"
USAGE_REPORTING = "usage_reporting"
LATENT_REASONING = "latent_reasoning"
HIDDEN_STATE_ACCESS = "hidden_state_access"
LATENT_READOUT = "latent_readout"
LATENT_EXIT_TRACE = "latent_exit_trace"
HYBRIDLINK_ACCESS = "hybridlink_access"

ALL_CAPABILITIES = (
    TEXT_GENERATION,
    STRUCTURED_GENERATION,
    TOOL_CALLS,
    USAGE_REPORTING,
    LATENT_REASONING,
    HIDDEN_STATE_ACCESS,
    LATENT_READOUT,
    LATENT_EXIT_TRACE,
    HYBRIDLINK_ACCESS,
)

LATENT_CAPABILITIES = {
    LATENT_REASONING,
    HIDDEN_STATE_ACCESS,
    LATENT_READOUT,
    LATENT_EXIT_TRACE,
    HYBRIDLINK_ACCESS,
}


@dataclass
class Capabilities:
    text_generation: bool = False
    structured_generation: bool = False
    tool_calls: bool = False
    usage_reporting: bool = False
    latent_reasoning: bool = False
    hidden_state_access: bool = False
    latent_readout: bool = False
    latent_exit_trace: bool = False
    hybridlink_access: bool = False

    def as_set(self) -> set[str]:
        out = set()
        for name in ALL_CAPABILITIES:
            if getattr(self, name, False):
                out.add(name)
        return out

    def supports(self, required: Iterable[str]) -> bool:
        have = self.as_set()
        return set(required).issubset(have)

    def to_dict(self) -> dict:
        return {name: getattr(self, name) for name in ALL_CAPABILITIES}
