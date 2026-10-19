# OpenCode Integration Report

## Summary

Implemented a provider abstraction with OpenCode hosted and native hybrid adapters, capability routing, configuration, CLI integration, experiment adapter, tests, and documentation.

## Compatibility matrix

| Feature           | OpenCode                       | Native HybridAgent       |
| ----------------- | ------------------------------ | ------------------------ |
| Text generation   | Supported when configured      | Supported                |
| Structured output | Model-dependent                | Implementation-dependent |
| Tool calls        | Endpoint/model-dependent       | Native tools             |
| Hidden states     | Not available via standard API | Available                |
| Latent reasoning  | Not available via standard API | Available                |
| HybridLink        | Not available via standard API | Available                |
| Dual readouts     | Not available via standard API | Requires instrumentation |

## Changed files

- hybrid_agent/__init__.py (lazy imports)
- hybrid_agent/providers/* (new)
- hybrid_agent/experiments/* (new)
- hybrid_inference/run_batch.py
- configs/opencode.toml
- docs/opencode-provider.md
- docs/opencode-integration-report.md
- tests/* (new)

## Tests

Offline tests in tests/ cover success, auth, missing creds, unknown model, rate limit, timeout, malformed, missing usage, structured output, capability refusal, retry exhaustion, budget, raw preservation, model identity, JSONL continuation/resume, native unaffected, latent rejection, no invented hidden states, arm mapping.

## Limitations

- Live API not executed.
- Native hybrid path unchanged but not re-run here.
- Cost enforcement relies on request counts unless pricing available.
