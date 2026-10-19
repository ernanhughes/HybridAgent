"""Provider errors."""

from __future__ import annotations


class ProviderError(Exception):
    """Base error for provider failures."""


class UnsupportedCapabilityError(ProviderError):
    """Raised when a request requires a capability the provider does not expose."""


class AuthenticationError(ProviderError):
    """Missing or invalid credentials."""


class RateLimitError(ProviderError):
    """Provider rate limit exceeded."""


class TimeoutError(ProviderError):  # noqa: A001 - intentional domain error
    """Request timed out."""


class ModelNotFoundError(ProviderError):
    """Requested model is not available."""


class MalformedResponseError(ProviderError):
    """Provider returned an unparseable response."""


class BudgetExhaustedError(ProviderError):
    """Request budget exhausted."""


class ConfigurationError(ProviderError):
    """Invalid provider configuration."""
