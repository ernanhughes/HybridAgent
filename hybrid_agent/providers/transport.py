"""HTTP transport abstraction (default: httpx, fallback: urllib)."""

from __future__ import annotations

import json as _json
from dataclasses import dataclass
from typing import Any, Callable

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore


@dataclass
class HttpResponse:
    status_code: int
    headers: dict
    data: Any
    text: str = ""


Transport = Callable[..., HttpResponse]


def default_transport(
    method: str,
    url: str,
    *,
    headers: dict | None = None,
    json: Any = None,
    timeout: float = 120.0,
) -> HttpResponse:
    if httpx is not None:
        with httpx.Client(timeout=timeout) as client:
            response = client.request(method, url, headers=headers or {}, json=json)
        try:
            data = response.json()
        except Exception:
            data = None
        return HttpResponse(
            status_code=response.status_code,
            headers=dict(response.headers),
            data=data,
            text=response.text,
        )

    import urllib.request
    import urllib.error

    payload = _json.dumps(json).encode("utf-8") if json is not None else None
    req = urllib.request.Request(url, data=payload, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            try:
                data = _json.loads(raw)
            except Exception:
                data = None
            return HttpResponse(
                status_code=getattr(response, "status", 200),
                headers=dict(response.headers),
                data=data,
                text=raw,
            )
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            data = _json.loads(raw)
        except Exception:
            data = None
        return HttpResponse(
            status_code=exc.code,
            headers=dict(exc.headers or {}),
            data=data,
            text=raw,
        )
    except Exception as exc:
        raise RuntimeError(f"transport error: {exc}") from exc
