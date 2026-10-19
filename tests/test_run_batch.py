import json


from hybrid_inference import run_batch


class FakeProvider:
    def __init__(self):
        self.calls = 0
        self.capabilities = type("C", (), {"to_dict": lambda self: {"text_generation": True}})()

    @property
    def provider_id(self):
        return "opencode"

    def generate(self, req):
        self.calls += 1
        from hybrid_agent.providers.base import ProviderResponse
        return ProviderResponse(text=f"response {self.calls}", model="m", provider="opencode", usage=None, raw={"x": 1}, latency_s=0.1)

    def check_config(self):
        return {"ok": True}

    def close(self):
        pass


def test_jsonl_continuation_and_resume(tmp_path, monkeypatch):
    inp = tmp_path / "in.jsonl"
    inp.write_text('{"id":"q1","question":"hi"}\n{"id":"q2","question":"bye"}\n', encoding="utf-8")
    out = tmp_path / "out.jsonl"

    monkeypatch.setenv("OPENCODE_API_KEY", "k")
    monkeypatch.setattr(run_batch, "create_provider", lambda cfg, **kw: FakeProvider())

    args = [
        "--provider", "opencode",
        "--model", "m",
        "--input", str(inp),
        "--output", str(out),
        "--num_samples", "2",
    ]
    run_batch.main(args)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4
    recs = [json.loads(line) for line in lines]
    assert all(r["status"] == "ok" for r in recs)
    assert all(r["execution_mode"] == "TEXT_API" for r in recs)

    # resume: should not duplicate
    run_batch.main(args + ["--resume"])
    lines2 = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines2) == 4


def test_budget_exhaustion(tmp_path, monkeypatch):
    inp = tmp_path / "in.jsonl"
    inp.write_text('{"id":"q1","question":"hi"}\n', encoding="utf-8")
    out = tmp_path / "out.jsonl"
    monkeypatch.setenv("OPENCODE_API_KEY", "k")
    monkeypatch.setattr(run_batch, "create_provider", lambda cfg, **kw: FakeProvider())
    args = [
        "--provider", "opencode",
        "--model", "m",
        "--input", str(inp),
        "--output", str(out),
        "--num_samples", "5",
        "--max-requests", "2",
    ]
    run_batch.main(args)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_dry_run(tmp_path, monkeypatch):
    inp = tmp_path / "in.jsonl"
    inp.write_text('{"id":"q1","question":"hi"}\n', encoding="utf-8")
    out = tmp_path / "out.jsonl"
    monkeypatch.setenv("OPENCODE_API_KEY", "k")
    monkeypatch.setattr(run_batch, "create_provider", lambda cfg, **kw: FakeProvider())
    args = [
        "--provider", "opencode",
        "--model", "m",
        "--input", str(inp),
        "--output", str(out),
        "--dry-run",
    ]
    run_batch.main(args)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    r = json.loads(lines[0])
    assert r["status"] == "dry_run"
    assert r["response"] is None


def test_check_config(tmp_path, monkeypatch, capsys):
    inp = tmp_path / "in.jsonl"
    inp.write_text('{"id":"q1","question":"hi"}\n', encoding="utf-8")
    out = tmp_path / "out.jsonl"
    monkeypatch.setenv("OPENCODE_API_KEY", "k")
    monkeypatch.setattr(run_batch, "create_provider", lambda cfg, **kw: FakeProvider())
    args = [
        "--provider", "opencode",
        "--model", "m",
        "--input", str(inp),
        "--output", str(out),
        "--check-config",
    ]
    run_batch.main(args)
    captured = capsys.readouterr()
    assert "ok" in captured.out
