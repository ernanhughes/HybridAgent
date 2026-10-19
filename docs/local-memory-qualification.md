# Local Memory Qualification (HybridAgent-8B)

## Hardware

- System RAM: 127.9 GiB
- GPU: NVIDIA GeForce RTX 2060
- Dedicated VRAM: 12 GiB (approx 9.2 GiB free at measurement)
- Host OS: Windows
- Driver: 591.44

## Memory budget (estimate)

For HybridAgent-8B with FP16:

| Component | Estimate |
|---|---|
| Backbone weights (FP16) | ~14.9 GiB |
| KV cache (4096 tokens) | ~9 GiB (worst case; less for short runs) |
| Readout bank (FP32) | ~2.3 GiB |
| Activations / latent feedback | ~1-2 GiB |
| GPU reserve | ~2 GiB |

System RAM must hold weights when offloaded (~15 GiB plus overhead). 128 GiB is sufficient.

## Recommended first configuration (FP16, partial offload)

```
python -m hybrid_inference.qualify_local `
  --model togethercomputer/HybridAgent-8B-Agentic `
  --dtype float16 `
  --device-map auto `
  --max-memory '{"0":"9GiB","cpu":"64GiB"}' `
  --offload-folder C:\Projects\HybridAgent\offload `
  --output outputs/qualify-local.json
```

This is an FP16 deviation from the original BF16 research configuration (RTX 2060 has no native BF16). Results are not scientifically equivalent to the released BF16 configuration.

## Fallbacks

1. FP16 CPU-only: `--dtype float16 --device cpu` (no device-map). Slow but compatible.
2. FP32 CPU-only: `--dtype float32 --device cpu`.
3. 8-bit/4-bit: `--load-in-8bit` or `--load-in-4bit` only if FP16 paths fail or are impractical.

## Scientific limitations

- Mixed CPU/GPU dispatch may alter numerical results.
- HybridAgent's custom recurrent loop, HybridLink, KV cache and readout bank require correct device handling; Accelerate dispatch alone does not guarantee compatibility.
- A successful one-item qualification does not authorize full experiment execution.
- No weights are downloaded or inference run by the unit tests.

## Opt-in changes

All memory/offloading options are explicit CLI flags. Default `load_hybrid_agent` behavior remains BF16 with optional `device` (original reference path).
