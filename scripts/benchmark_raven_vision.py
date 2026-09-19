from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from PIL import Image

from arenaagent.preliminary_baseline_agent.tasks.raven.solver import HybridRavenSolver
from arenaagent.vlm_agent.client import ClientFactory
from arenaagent.vlm_agent.raven_skill import normalize_raven_image_list, run_raven_inference
from arenaagent.vlm_agent.vlm_config import VLMClientCfg

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLES = [
    ("logs/raven_task_data/265864fc0f_8pac0l4g/task_data_001.png", [2, 2, 6]),
    ("logs/raven_task_data/6d08e3d112_v00ypi86/task_data_001.png", [5, 8, 2]),
    ("logs/raven_task_data/f2ee8b293a_4ibkew1a/task_data_001.png", [3, 6, 6]),
    ("logs/raven_task_data/9158b815d9_40en0g6j/task_data_001.png", [2, 2, 3]),
]


def build_client(model: str, timeout: float) -> Any:
    cfg = VLMClientCfg()
    cfg.name = model
    cfg.api_base = os.environ["VLM_CLIENT_CFG_API_BASE"]
    cfg.api_key = os.environ["VLM_CLIENT_CFG_API_KEY"]
    cfg.message_role = "user"
    cfg.request_timeout_seconds = timeout
    cfg.native_max_retries = 0
    if model in {"kimi-k3", "kimi-k2.6"}:
        cfg.chat_completion_kwargs = {
            "extra_body": {"thinking": {"type": "disabled"}},
            "max_tokens": 256 if model == "kimi-k3" else 1800,
        }
    return ClientFactory().build("openai", cfg)


def benchmark_model(model: str, samples: list[tuple[Path, list[int]]], timeout: float) -> dict[str, Any]:
    client = build_client(model, timeout)
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for image_path, expected in samples:
        image_groups = normalize_raven_image_list(str(image_path))
        if not image_groups:
            results.append({"image": str(image_path), "error": "crop failed"})
            continue
        legacy = run_raven_inference(image_groups, []) or []
        sample_started = time.perf_counter()
        solved = HybridRavenSolver(vlm_client=client, text_client=None).solve(
            image_groups=image_groups,
            legacy_ranked=legacy,
            whole_image=Image.open(image_path).convert("RGB"),
        )
        elapsed = time.perf_counter() - sample_started
        visual_by_question = {
            int(vote["question"]): int(vote["answer"])
            for vote in solved.diagnostics.get("visual_votes", [])
        }
        visual = [visual_by_question.get(index) for index in range(1, 4)]
        selected = solved.selected_answers
        results.append(
            {
                "image": image_path.name,
                "expected": expected,
                "legacy": legacy[0] if legacy else None,
                "visual": visual,
                "fused": selected,
                "visual_correct": sum(a == b for a, b in zip(visual, expected)),
                "fused_correct": sum(a == b for a, b in zip(selected, expected)),
                "exact": selected == expected,
                "elapsed_seconds": round(elapsed, 3),
            }
        )
        print(json.dumps({"model": model, "latest": results[-1]}, ensure_ascii=False), flush=True)
    return {
        "model": model,
        "results": results,
        "visual_digits_correct": sum(item.get("visual_correct", 0) for item in results),
        "fused_digits_correct": sum(item.get("fused_correct", 0) for item in results),
        "total_digits": 3 * len(results),
        "exact_subjects": sum(bool(item.get("exact")) for item in results),
        "total_seconds": round(time.perf_counter() - started, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["kimi-k3", "kimi-k2.6"])
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--sample", type=int, choices=range(len(DEFAULT_SAMPLES)))
    args = parser.parse_args()
    load_dotenv(ROOT / ".env", override=False)
    missing = [name for name in ("VLM_CLIENT_CFG_API_BASE", "VLM_CLIENT_CFG_API_KEY") if not os.getenv(name)]
    if missing:
        raise SystemExit("Missing required configuration: " + ", ".join(missing))
    samples = [(ROOT / relative_path, expected) for relative_path, expected in DEFAULT_SAMPLES]
    if args.sample is not None:
        samples = [samples[args.sample]]
    summaries = [benchmark_model(model, samples, args.timeout) for model in args.models]
    print("BENCHMARK_SUMMARY=" + json.dumps(summaries, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
