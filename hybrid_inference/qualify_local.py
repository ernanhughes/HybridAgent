"""Local HybridAgent qualification harness (bounded, one item)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from hybrid_agent import HybridGenerationEngine, SYSTEM_PROMPT, load_hybrid_agent
from hybrid_agent.tools import ToolExecutor


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="HybridAgent local qualification")
    p.add_argument("--model", required=True)
    p.add_argument("--input", default=None, help="JSONL; first question used")
    p.add_argument("--question", default="How many trailing zeros does 100! have?")
    p.add_argument("--output", default="outputs/qualify-local.json")
    p.add_argument("--dtype", default="float16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default=None)
    p.add_argument("--device-map", default=None)
    p.add_argument("--offload-folder", default=None)
    p.add_argument("--max-memory", default=None, help="JSON dict, e.g. '{\"0\":\"9GiB\",\"cpu\":\"64GiB\"}'")
    p.add_argument("--load-in-8bit", action="store_true")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--max-response-tokens", type=int, default=1024)
    p.add_argument("--min-latent-steps", type=int, default=4)
    p.add_argument("--max-latent-steps", type=int, default=64)
    p.add_argument("--estimate-memory", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    if args.max_memory:
        args.max_memory = json.loads(args.max_memory)
    return args


def estimate_memory(args):
    params_b = 8.0
    bytes_per = {"bfloat16": 2, "float16": 2, "float32": 4}[args.dtype]
    weights_gb = params_b * 1e9 * bytes_per / (1024 ** 3)
    # rough KV cache for 4096 tokens, 36 layers, hidden 4096, 2 (k+v)
    kv_gb = (2 * 36 * 4096 * 4096 * bytes_per * 4096) / (1024 ** 3) / 1024  # crude
    readout_gb = (151936 * 4096 * 4) / (1024 ** 3)
    gpu_reserve_gb = 2.0
    return {
        "dtype": args.dtype,
        "weights_gb": round(weights_gb, 2),
        "kv_cache_gb_est": round(kv_gb, 2),
        "readout_bank_gb_est": round(readout_gb, 2),
        "gpu_reserve_gb": gpu_reserve_gb,
        "system_ram_recommended_gb": round(weights_gb + 8, 2),
        "note": "Estimates only; actual depends on sequence length, batch=1, and offloading.",
    }


def _pick_question(args):
    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                return rec.get("question", args.question)
    return args.question


def main(argv=None):
    args = parse_args(argv)

    if args.estimate_memory or args.dry_run:
        est = estimate_memory(args)
        print(json.dumps(est, indent=2))
        return

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    device = args.device
    device_map = args.device_map
    if args.load_in_8bit or args.load_in_4bit:
        device_map = device_map or "auto"

    t_load = time.time()
    model, tokenizer = load_hybrid_agent(
        args.model,
        dtype=args.dtype,
        device=device if device_map is None else None,
        device_map=device_map,
        offload_folder=args.offload_folder,
        max_memory=args.max_memory,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )
    load_s = time.time() - t_load

    question = _pick_question(args)
    engine_device = getattr(model, "device", None)
    try:
        engine_device = model.backbone.get_input_embeddings().weight.device
    except Exception:
        pass
    if engine_device is None:
        engine_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tools = ToolExecutor()
    engine = HybridGenerationEngine(
        model, tokenizer, engine_device, tools, SYSTEM_PROMPT, question,
        batch_size=1,
        max_response_tokens=args.max_response_tokens,
        max_tool_calls=0,
        max_search_calls=0,
        max_python_calls=0,
        min_latent_steps=args.min_latent_steps,
        max_latent_steps=args.max_latent_steps,
        record_latent_readout=True,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    peak_ram_before = None
    try:
        import psutil  # type: ignore
        peak_ram_before = psutil.Process().memory_info().rss
    except Exception:
        pass

    rng = torch.Generator(device=engine_device)
    rng.manual_seed(args.seed)
    t0 = time.time()
    failure = None
    try:
        engine.run(rng)
        status = "ok"
    except Exception as exc:
        status = "error"
        failure = f"{type(exc).__name__}: {exc}"
    wall = time.time() - t0

    result = None
    if status == "ok":
        result = engine.row_result(engine.rows[0])

    peak_gpu = None
    if torch.cuda.is_available():
        peak_gpu = torch.cuda.max_memory_allocated()
    peak_ram = None
    try:
        import psutil  # type: ignore
        peak_ram = psutil.Process().memory_info().rss
    except Exception:
        pass

    out = {
        "status": status,
        "failure": failure,
        "model": args.model,
        "dtype": args.dtype,
        "device_map": device_map,
        "load_time_s": round(load_s, 2),
        "wall_time_s": round(wall, 2),
        "engine_device": str(engine_device),
        "latent_entry": bool(result and result.get("latent_spans", 0) > 0) if result else False,
        "latent_steps": result.get("latent_steps") if result else None,
        "latent_spans": result.get("latent_spans") if result else None,
        "anchor_exits": result.get("anchor_exits") if result else None,
        "forced_exits": result.get("forced_exits") if result else None,
        "end_reason": result.get("end_reason") if result else None,
        "latent_readouts": result.get("latent_readouts") if result else None,
        "peak_gpu_bytes": peak_gpu,
        "peak_ram_bytes": peak_ram,
        "peak_ram_delta_bytes": (peak_ram - peak_ram_before) if (peak_ram and peak_ram_before) else None,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
