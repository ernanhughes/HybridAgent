"""Batch inference with HybridAgent: reads questions from a JSONL file, runs hybrid
generation with the Python and search tools (several samples of one question share
one working-memory prefill), and appends one JSON record per sample with the output,
the extracted answer, the latent-span readouts and tool-call statistics.

  python -m hybrid_inference.run_batch --model <checkpoint dir or HF id> \
      --input examples/questions.jsonl --output outputs/results.jsonl --num_samples 4

OpenCode text provider:

  python -m hybrid_inference.run_batch --provider opencode --model <model-id> \
      --input examples/questions.jsonl --output outputs/opencode-results.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

from hybrid_agent.providers import (
    create_provider,
    load_config,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="HybridAgent batch inference")
    p.add_argument("--provider", default="native", help="provider: native or opencode")
    p.add_argument("--config", default=None, help="path to provider config file")
    p.add_argument("--model", default=None, help="model id or checkpoint (required for native)")
    p.add_argument("--base-url", default=None, help="override OpenCode base URL")
    p.add_argument("--timeout", type=float, default=None, help="override timeout seconds")
    p.add_argument("--max-retries", type=int, default=None, help="override max retries")
    p.add_argument("--input", required=True, help="JSONL with one {\"id\", \"question\"} per line")
    p.add_argument("--output", required=True, help="JSONL output (appended; see --resume)")
    p.add_argument("--num_samples", type=int, default=1, help="samples per question")
    p.add_argument("--batch_size", type=int, default=None, help="samples generated together (default: num_samples)")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--max_response_tokens", type=int, default=4096, help="text tokens + latent steps per sample")
    p.add_argument("--count_tool_tokens", action="store_true", help="count tool observations toward the budget")
    p.add_argument("--max_tool_calls", type=int, default=5, help="tool calls per sample")
    p.add_argument("--max_search_calls", type=int, default=5)
    p.add_argument("--max_python_calls", type=int, default=5)
    p.add_argument("--continue_on_tool_cap", action="store_true",
                   help="keep generating after the tool cap instead of ending the sample")
    p.add_argument("--max_turns", type=int, default=30)
    p.add_argument("--min_latent_steps", type=int, default=4)
    p.add_argument("--max_latent_steps", type=int, default=200)
    p.add_argument("--no_latent_readout", action="store_true", help="skip the pseudo-text readout of latent spans")
    p.add_argument("--search_max_results", type=int, default=10)
    p.add_argument("--search_result_length", type=int, default=500)
    p.add_argument("--python_timeout", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shard_id", type=int, default=None, help="process this shard of the questions (default: RANK env)")
    p.add_argument("--num_shards", type=int, default=None, help="total shards (default: WORLD_SIZE env)")
    p.add_argument("--resume", action="store_true", help="skip (id, sample) pairs already in --output")
    p.add_argument("--limit", type=int, default=0, help="stop after this many questions (0 = all)")
    p.add_argument("--dry-run", action="store_true", help="do not contact the API; write dry-run records")
    p.add_argument("--check-config", action="store_true", help="print provider configuration and exit")
    p.add_argument("--max-requests", type=int, default=None, help="override execution max_requests")
    p.add_argument("--max-cost-usd", type=float, default=None, help="override execution max_cost_usd")
    args = p.parse_args(argv)
    if args.batch_size is None:
        args.batch_size = args.num_samples
    if args.shard_id is None:
        args.shard_id = int(os.environ.get("RANK", 0))
    if args.num_shards is None:
        args.num_shards = int(os.environ.get("WORLD_SIZE", 1))
    return args


def read_questions(path: str) -> list[dict]:
    questions: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "question" not in rec:
                raise ValueError(f"record without a question: {line[:80]}")
            rec.setdefault("id", len(questions))
            questions.append(rec)
    return questions


def read_done(path: str) -> set[tuple]:
    done = set()
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add((rec["id"], rec["sample_idx"]))
                except Exception:
                    continue
    return done


def _load_provider_config(args):
    cfg = load_config(args.config)
    cfg.provider = args.provider
    if args.provider in ("opencode", "opencode-go"):
        if args.model:
            cfg.opencode.model = args.model
        if args.base_url:
            cfg.opencode.base_url = args.base_url
        if args.timeout is not None:
            cfg.opencode.timeout_seconds = args.timeout
        if args.max_retries is not None:
            cfg.opencode.max_retries = args.max_retries
    else:
        if args.model:
            cfg.native_hybrid.model = args.model
            cfg.native_hybrid.enabled = True
    if args.max_requests is not None:
        cfg.execution.max_requests = args.max_requests
    if args.max_cost_usd is not None:
        cfg.execution.max_cost_usd = args.max_cost_usd
    if args.dry_run:
        cfg.execution.dry_run = True
    cfg.validate()
    return cfg


def _run_opencode(args) -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    cfg = _load_provider_config(args)
    provider = create_provider(cfg)

    if args.check_config:
        import pprint

        pprint.pprint(provider.check_config())
        return

    questions: list[dict] = read_questions(args.input)
    if args.limit:
        questions = questions[: args.limit]
    mine = [(i, q) for i, q in enumerate(questions) if i % args.num_shards == args.shard_id]
    done = read_done(args.output) if args.resume else set()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    run_id = str(uuid.uuid4())
    request_count = 0
    max_requests = cfg.execution.max_requests
    # max_cost = cfg.execution.max_cost_usd  # cost enforcement not implemented (unknown pricing)

    print(f"[shard {args.shard_id}/{args.num_shards}] provider=opencode model={cfg.opencode.model} "
          f"{len(mine)} questions x {args.num_samples} samples -> {args.output}", flush=True)

    with open(args.output, "a", encoding="utf-8") as out:
        for q_index, rec in mine:
            pending = [s for s in range(args.num_samples) if (rec["id"], s) not in done]
            if not pending:
                continue
            qhash = hashlib.sha256(rec["question"].encode("utf-8")).hexdigest()
            meta = {k: v for k, v in rec.items() if k not in ("id", "question")}
            for sample_idx in pending:
                if request_count >= max_requests:
                    print("request budget exhausted; stopping", flush=True)
                    return
                record = {
                    "run_id": run_id,
                    "id": rec["id"],
                    "sample_idx": sample_idx,
                    "question": rec["question"],
                    "input_hash": qhash,
                    "provider": provider.provider_id,
                    "model": cfg.opencode.model,
                    "execution_mode": "TEXT_API",
                    "capability_flags": provider.capabilities.to_dict() if hasattr(provider.capabilities, "to_dict") else provider.capabilities.as_set() if hasattr(provider.capabilities, "as_set") else {},
                    "meta": meta if meta else None,
                }
                if cfg.execution.dry_run:
                    record.update({
                        "status": "dry_run",
                        "response": None,
                        "raw_response": None,
                        "usage": None,
                        "latency_s": 0.0,
                        "error": None,
                    })
                else:
                    from hybrid_agent.providers.base import ProviderRequest

                    messages = [{"role": "user", "content": rec["question"]}]
                    req = ProviderRequest(
                        messages=messages,
                        model=cfg.opencode.model,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_response_tokens,
                    )
                    t0 = time.time()
                    try:
                        resp = provider.generate(req)
                        record.update({
                            "status": "ok",
                            "response": resp.text,
                            "raw_response": resp.raw if cfg.execution.preserve_raw else None,
                            "usage": resp.usage.to_dict() if resp.usage else None,
                            "latency_s": resp.latency_s,
                            "model_reported": resp.model,
                            "request_id": resp.request_id,
                            "error": None,
                        })
                    except Exception as exc:
                        record.update({
                            "status": "error",
                            "response": None,
                            "raw_response": None,
                            "usage": None,
                            "latency_s": round(time.time() - t0, 4),
                            "error": f"{type(exc).__name__}: {exc}",
                        })
                request_count += 1
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                os.fsync(out.fileno())
                status = record.get('status')
                msg = f"[shard {args.shard_id}] question {rec['id']} sample {sample_idx}: {status}"
                if status == 'error':
                    msg += f" - {record.get('error')}"
                print(msg, flush=True)

    provider.close()


def _run_native(args) -> None:
    import torch

    from hybrid_agent import (
        HybridGenerationEngine,
        SYSTEM_PROMPT,
        build_tool_executor,
        extract_answer,
        load_hybrid_agent,
    )

    args_model = args.model
    if not args_model:
        raise SystemExit("--model is required for native provider")

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_hybrid_agent(args_model, device=device)
    tools = build_tool_executor(args.python_timeout, args.search_max_results, args.search_result_length)

    questions: list[dict] = read_questions(args.input)
    if args.limit:
        questions = questions[: args.limit]
    mine = [(i, q) for i, q in enumerate(questions) if i % args.num_shards == args.shard_id]
    done = read_done(args.output) if args.resume else set()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    print(f"[shard {args.shard_id}/{args.num_shards}] {len(mine)} questions x {args.num_samples} samples -> {args.output}", flush=True)

    with open(args.output, "a", encoding="utf-8") as out:
        for q_index, rec in mine:
            pending = [s for s in range(args.num_samples) if (rec["id"], s) not in done]
            if not pending:
                continue
            meta = {k: v for k, v in rec.items() if k not in ("id", "question")}
            for start in range(0, len(pending), args.batch_size):
                sample_ids = pending[start: start + args.batch_size]
                engine = HybridGenerationEngine(
                    model, tokenizer, device, tools, SYSTEM_PROMPT, rec["question"],
                    batch_size=len(sample_ids),
                    max_response_tokens=args.max_response_tokens,
                    count_tool_tokens=args.count_tool_tokens,
                    max_tool_calls=args.max_tool_calls,
                    max_search_calls=args.max_search_calls,
                    max_python_calls=args.max_python_calls,
                    terminate_on_tool_cap=not args.continue_on_tool_cap,
                    max_turns=args.max_turns,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    min_latent_steps=args.min_latent_steps,
                    max_latent_steps=args.max_latent_steps,
                    record_latent_readout=not args.no_latent_readout,
                )
                rng = torch.Generator(device=device)
                rng.manual_seed(args.seed * 1_000_003 + q_index * 1_000 + sample_ids[0])
                t0 = time.time()
                engine.run(rng)
                wall = time.time() - t0
                for row, sample_idx in zip(engine.rows, sample_ids):
                    result = engine.row_result(row)
                    record = {
                        "id": rec["id"],
                        "sample_idx": sample_idx,
                        "question": rec["question"],
                        **({"meta": meta} if meta else {}),
                        "output": result["output"],
                        "prediction": extract_answer(result["output"]),
                        "latent_readouts": result.get("latent_readouts", []),
                        "output_rendered": result.get("output_rendered", result["output"]),
                        "num_tool_calls": result["python_calls"] + result["search_calls"],
                        "num_search_calls": result["search_calls"],
                        "num_python_calls": result["python_calls"],
                        "generation_steps": result["generation_steps"],
                        "latent_steps": result["latent_steps"],
                        "latent_spans": result["latent_spans"],
                        "end_reason": result["end_reason"],
                        "wall_time_s": round(wall, 2),
                        "execution_mode": "NATIVE_HYBRID",
                        "provider": "native_hybrid",
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                os.fsync(out.fileno())
                print(f"[shard {args.shard_id}] question {rec['id']}: {len(sample_ids)} samples in {wall:.0f}s "
                      f"(latent steps {sum(engine.row_result(r)['latent_steps'] for r in engine.rows)})", flush=True)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.provider in ("opencode", "opencode-go"):
        _run_opencode(args)
    else:
        _run_native(args)


if __name__ == "__main__":
    main()
