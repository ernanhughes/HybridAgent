# OpenCode Provider for HybridAgent

## Architecture

```
HybridAgent Experiment Runner
        |
        +-- OpenCode Provider (hosted)  --> OpenCode Inference API (text generation)
        |
        +-- Native Hybrid Provider       --> Model weights + HybridLink (latent reasoning)
```

OpenCode-hosted inference executes ordinary model API requests. It cannot reproduce the latent reasoning of HybridAgent unless the endpoint exposes the required model-level operations. Consequently, hosted text results are controls or baselines, not latent reasoning observations.

## Installation

Lightweight path (no CUDA):

```
python -m pip install httpx pyyaml python-dotenv
```

Native path additionally requires torch/transformers (see requirements.txt).

## Credential setup

```
$env:OPENCODE_API_KEY = "your-api-key"
```

Never commit keys. `.env` is gitignored.

## Model discovery

Inspect available models via the OpenCode catalog. Do not hard-code model IDs.

## Configuration example

`configs/opencode.toml`:

```toml
provider = "opencode"

[opencode]
api_family = "openai-chat"
base_url = "https://opencode.ai/inference/openai/v1"
model = "YOUR_MODEL_ID"
api_key_env = "OPENCODE_API_KEY"
timeout_seconds = 120
max_retries = 2
supports_structured_output = false
supports_tool_calls = false

[execution]
preserve_raw = true
dry_run = false
max_requests = 100
max_cost_usd = 2.00
```

## Windows usage

```
cd C:\Projects\HybridAgent
$env:OPENCODE_API_KEY = "your-api-key"
python -m hybrid_inference.run_batch --provider opencode --model YOUR_MODEL_ID --config configs/opencode.toml --input examples/questions.jsonl --output outputs/opencode-results.jsonl
```

## Batch command

```
python -m hybrid_inference.run_batch --provider opencode --model YOUR_MODEL_ID --input examples/questions.jsonl --output outputs/opencode-results.jsonl
```

## Dry run

```
python -m hybrid_inference.run_batch --provider opencode --model YOUR_MODEL_ID --input examples/questions.jsonl --output outputs/dry.jsonl --dry-run
```

Configuration check without network:

```
python -m hybrid_inference.run_batch --provider opencode --model YOUR_MODEL_ID --input examples/questions.jsonl --output outputs/x.jsonl --check-config
```

## Optional live smoke test

A live call is authorized only if credentials are available, endpoint/model verified, small output limit, request budget configured, and explicit user authorization. Never print API keys.

## Language experiment mapping

OpenCode may be used for arms B, C, recipient execution, Z, and X. It must not be used for latent arms D or E. The adapter is `hybrid_agent.experiments.lang_latent_01`.

## Capability restrictions

Hosted OpenCode supports text generation (when configured). Structured output and tool calls are model/endpoint dependent. Hidden states, latent reasoning, HybridLink, dual readouts, and latent exit traces are not available via the standard API.

## Native versus hosted comparison

| Feature           | OpenCode                       | Native HybridAgent       |
| ----------------- | ------------------------------ | ------------------------ |
| Text generation   | Supported when configured      | Supported                |
| Structured output | Model-dependent                | Implementation-dependent |
| Tool calls        | Endpoint/model-dependent       | Native tools             |
| Hidden states     | Not available via standard API | Available                |
| Latent reasoning  | Not available via standard API | Available                |
| HybridLink        | Not available via standard API | Available                |
| Dual readouts     | Not available via standard API | Requires instrumentation |

## Known limitations

- No hidden-state access.
- Cost enforcement requires pricing/usage; request-count limits are enforced instead when cost unknown.
- No scientific experiment is authorized by this adapter.

## Troubleshooting

- Missing credentials: set OPENCODE_API_KEY.
- Unknown model: verify model ID from catalog.
- Rate limits: reduce concurrency or configure retries.
- Import errors: ensure lightweight deps installed.
