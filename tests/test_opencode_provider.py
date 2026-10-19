import json

import pytest

from hybrid_agent.providers import (
    OpenCodeProvider,
    ProviderConfig,
    ProviderRequest,
    UnsupportedCapabilityError,
    AuthenticationError,
    RateLimitError,
    ModelNotFoundError,
    MalformedResponseError,
)
from hybrid_agent.providers.capabilities import Capabilities


def make_provider(transport, model="m1", **cfg_overrides):
    pcfg = ProviderConfig(
        provider="opencode",
        model=model,
        base_url="https://example.com/v1",
        api_key_env="OPENCODE_API_KEY_TEST",
        timeout_seconds=5.0,
        max_retries=cfg_overrides.get("max_retries", 2),
    )
    from hybrid_agent.providers.config import OpenCodeConfig
    oc = OpenCodeConfig(
        base_url="https://example.com/v1",
        model=model,
        api_key_env="OPENCODE_API_KEY_TEST",
        timeout_seconds=5.0,
        max_retries=cfg_overrides.get("max_retries", 2),
        supports_structured_output=cfg_overrides.get("structured", False),
        supports_tool_calls=cfg_overrides.get("tools", False),
    )
    return OpenCodeProvider(pcfg, opencode_config=oc, transport=transport)


def ok_response(text="hello", model="m1", usage=None):
    body = {"choices": [{"message": {"content": text}, "finish_reason": "stop"}], "model": model}
    if usage is not None:
        body["usage"] = usage
    return type("R", (), {"status_code": 200, "headers": {}, "data": body, "text": json.dumps(body)})()


@pytest.fixture(autouse=True)
def env_key(monkeypatch):
    monkeypatch.setenv("OPENCODE_API_KEY_TEST", "secret")


def test_success_text():
    def transport(method, url, headers=None, json=None, timeout=None):
        return ok_response("hi there", usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5})
    p = make_provider(transport)
    req = ProviderRequest(messages=[{"role": "user", "content": "q"}])
    resp = p.generate(req)
    assert resp.text == "hi there"
    assert resp.usage.total_tokens == 5
    assert resp.provider == "opencode"
    assert resp.execution_mode == "TEXT_API"


def test_missing_credentials(monkeypatch):
    monkeypatch.delenv("OPENCODE_API_KEY_TEST", raising=False)
    def transport(*a, **k):
        raise AssertionError("should not call")
    p = make_provider(transport)
    req = ProviderRequest(messages=[{"role": "user", "content": "q"}])
    with pytest.raises(AuthenticationError):
        p.generate(req)


def test_auth_error():
    def transport(method, url, headers=None, json=None, timeout=None):
        return type("R", (), {"status_code": 401, "headers": {}, "data": {}, "text": "unauth"})()
    p = make_provider(transport)
    req = ProviderRequest(messages=[{"role": "user", "content": "q"}])
    with pytest.raises(AuthenticationError):
        p.generate(req)


def test_unknown_model():
    def transport(method, url, headers=None, json=None, timeout=None):
        return type("R", (), {"status_code": 404, "headers": {}, "data": {}, "text": "not found"})()
    p = make_provider(transport)
    req = ProviderRequest(messages=[{"role": "user", "content": "q"}])
    with pytest.raises(ModelNotFoundError):
        p.generate(req)


def test_rate_limit_exhaustion():
    calls = {"n": 0}
    def transport(method, url, headers=None, json=None, timeout=None):
        calls["n"] += 1
        return type("R", (), {"status_code": 429, "headers": {}, "data": {}, "text": "rl"})()
    p = make_provider(transport, max_retries=1)
    req = ProviderRequest(messages=[{"role": "user", "content": "q"}])
    with pytest.raises(RateLimitError):
        p.generate(req)
    assert calls["n"] == 2


def test_timeout():
    def transport(method, url, headers=None, json=None, timeout=None):
        raise TimeoutError("timed out")
    p = make_provider(transport, max_retries=0)
    req = ProviderRequest(messages=[{"role": "user", "content": "q"}])
    from hybrid_agent.providers.errors import TimeoutError as PTimeout
    with pytest.raises(PTimeout):
        p.generate(req)


def test_malformed_response():
    def transport(method, url, headers=None, json=None, timeout=None):
        return type("R", (), {"status_code": 200, "headers": {}, "data": "not dict", "text": "x"})()
    p = make_provider(transport)
    req = ProviderRequest(messages=[{"role": "user", "content": "q"}])
    with pytest.raises(MalformedResponseError):
        p.generate(req)


def test_missing_usage():
    def transport(method, url, headers=None, json=None, timeout=None):
        return ok_response("hi", usage=None)
    p = make_provider(transport)
    req = ProviderRequest(messages=[{"role": "user", "content": "q"}])
    resp = p.generate(req)
    assert resp.usage is None


def test_structured_output():
    def transport(method, url, headers=None, json=None, timeout=None):
        assert json.get("response_format") == {"type": "json_object"}
        return ok_response('{"a":1}')
    p = make_provider(transport, structured=True)
    req = ProviderRequest(messages=[{"role": "user", "content": "q"}], response_format={"type": "json_object"})
    resp = p.generate(req)
    assert resp.text == '{"a":1}'


def test_capability_refusal():
    def transport(*a, **k):
        raise AssertionError("should not call")
    p = make_provider(transport, structured=False)
    req = ProviderRequest(messages=[{"role": "user", "content": "q"}], response_format={"type": "json_object"})
    with pytest.raises(UnsupportedCapabilityError):
        p.generate(req)


def test_latent_rejected():
    def transport(*a, **k):
        raise AssertionError("should not call")
    p = make_provider(transport)
    req = ProviderRequest(messages=[{"role": "user", "content": "q"}], required_capabilities={"latent_reasoning"})
    with pytest.raises(UnsupportedCapabilityError):
        p.generate(req)


def test_retry_exhaustion():
    calls = {"n": 0}
    def transport(method, url, headers=None, json=None, timeout=None):
        calls["n"] += 1
        return type("R", (), {"status_code": 500, "headers": {}, "data": {}, "text": "err"})()
    p = make_provider(transport, max_retries=2)
    req = ProviderRequest(messages=[{"role": "user", "content": "q"}])
    from hybrid_agent.providers.errors import ProviderError
    with pytest.raises(ProviderError):
        p.generate(req)
    assert calls["n"] == 3


def test_raw_preservation():
    def transport(method, url, headers=None, json=None, timeout=None):
        return ok_response("hi")
    p = make_provider(transport)
    req = ProviderRequest(messages=[{"role": "user", "content": "q"}])
    resp = p.generate(req)
    assert isinstance(resp.raw, dict)
    assert resp.raw["choices"][0]["message"]["content"] == "hi"


def test_model_identity_persistence():
    def transport(method, url, headers=None, json=None, timeout=None):
        return ok_response("hi", model="reported-model")
    p = make_provider(transport, model="configured-model")
    req = ProviderRequest(messages=[{"role": "user", "content": "q"}])
    resp = p.generate(req)
    assert resp.model == "reported-model"


def test_check_config_no_network():
    def transport(*a, **k):
        raise AssertionError("no network")
    p = make_provider(transport)
    info = p.check_config()
    assert info["model_configured"] is True
    assert "api_key_present" in info


def test_no_hidden_state_fields():
    def transport(method, url, headers=None, json=None, timeout=None):
        return ok_response("hi")
    p = make_provider(transport)
    req = ProviderRequest(messages=[{"role": "user", "content": "q"}])
    resp = p.generate(req)
    d = resp.to_dict()
    for bad in ("hidden_states", "latent_spans", "latent_readouts", "exit_decisions"):
        assert bad not in d
        assert bad not in (resp.raw or {})


def test_arm_mapping():
    from hybrid_agent.experiments.lang_latent_01 import LangLatent01Adapter, LATENT_ARMS
    from hybrid_agent.providers.errors import UnsupportedCapabilityError

    class Dummy:
        capabilities = Capabilities()

    adapter = LangLatent01Adapter(provider=Dummy(), dry_run=True)
    for arm in ["B", "C", "Z", "X", "RECIPIENT"]:
        adapter.ensure_arm_allowed(arm)
    for arm in LATENT_ARMS:
        with pytest.raises(UnsupportedCapabilityError):
            adapter.ensure_arm_allowed(arm)
    # build requests
    r = adapter.build_b_request("src")
    assert r.messages
    r2 = adapter.build_recipient_request(None, "q", "vocab", format_hint="STATE_READOUT")
    assert "Representation" in r2.messages[1]["content"]
