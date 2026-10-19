"""Narrow adapter for LANG-LATENT-01 (read-only consumption of its contract).

OpenCode may be used for textual/typed mediation arms and recipient execution.
It must NOT be used as a substitute for latent arms D/E.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from hybrid_agent.providers.base import ProviderRequest, ProviderResponse
from hybrid_agent.providers.errors import UnsupportedCapabilityError


ALLOWED_ARMS = {"B", "C", "Z", "X", "RECIPIENT"}
LATENT_ARMS = {"D", "E", "D_PRIME", "X_D"}


@dataclass
class LangLatent01Adapter:
    provider: Any
    dry_run: bool = True

    def refused_arms(self) -> set[str]:
        return set(LATENT_ARMS)

    def ensure_arm_allowed(self, arm: str) -> None:
        arm = arm.upper()
        if arm in LATENT_ARMS:
            raise UnsupportedCapabilityError(
                f"Arm {arm} requires native hybrid latent capabilities; "
                "OpenCode text provider cannot substitute for it."
            )
        if arm not in ALLOWED_ARMS:
            raise ValueError(f"unknown arm: {arm}")

    def build_b_request(self, source_text: str, *, model: str | None = None) -> ProviderRequest:
        messages = [
            {"role": "system", "content": "You produce a concise, faithful prose summary of the supplied record."},
            {"role": "user", "content": source_text},
        ]
        return ProviderRequest(messages=messages, model=model, temperature=0.2, max_tokens=512)

    def build_c_request(self, source_text: str, *, model: str | None = None) -> ProviderRequest:
        messages = [
            {"role": "system", "content": "You produce a typed record (JSON) summarising the supplied record."},
            {"role": "user", "content": source_text},
        ]
        return ProviderRequest(
            messages=messages,
            model=model,
            temperature=0.2,
            max_tokens=512,
            response_format={"type": "json_object"},
        )

    def build_recipient_request(
        self,
        representation: str | None,
        question: str,
        vocabulary: str,
        *,
        format_hint: str = "ORDINARY_TEXT",
        model: str | None = None,
    ) -> ProviderRequest:
        content = (
            f"Vocabulary:\n{vocabulary}\n\nQuestion: {question}\n\n"
            f"Representation ({format_hint}):\n{representation if representation is not None else '(none)'}\n"
        )
        messages = [
            {"role": "system", "content": "You answer the question using only the supplied representation and vocabulary."},
            {"role": "user", "content": content},
        ]
        return ProviderRequest(messages=messages, model=model, temperature=0.0, max_tokens=256)

    def generate_b(self, source_text: str, *, model: str | None = None) -> str | None:
        self.ensure_arm_allowed("B")
        if self.dry_run:
            return None
        resp: ProviderResponse = self.provider.generate(self.build_b_request(source_text, model=model))
        return resp.text

    def generate_c(self, source_text: str, *, model: str | None = None) -> str | None:
        self.ensure_arm_allowed("C")
        if self.dry_run:
            return None
        resp: ProviderResponse = self.provider.generate(self.build_c_request(source_text, model=model))
        return resp.text

    def run_recipient(
        self,
        representation: str | None,
        question: str,
        vocabulary: str,
        *,
        format_hint: str = "ORDINARY_TEXT",
        model: str | None = None,
    ) -> str | None:
        self.ensure_arm_allowed("RECIPIENT")
        if self.dry_run:
            return None
        resp: ProviderResponse = self.provider.generate(
            self.build_recipient_request(representation, question, vocabulary, format_hint=format_hint, model=model)
        )
        return resp.text

    @staticmethod
    def content_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
