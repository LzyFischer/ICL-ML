#!/usr/bin/env python
"""Run one ICL experiment for one model on one or more tasks.

Examples
--------
# How does accuracy scale with the number of demonstrations?
python scripts/run_experiment.py --model Qwen/Qwen2.5-0.5B --task sst2 ag_news --exp shots

# Do the labels in the demonstrations matter? (gold / random / flipped / abstract)
python scripts/run_experiment.py --model Qwen/Qwen2.5-1.5B --task sst2 --exp labels --k 16

# Order sensitivity: 3 demo sets x 10 permutations each
python scripts/run_experiment.py --model Qwen/Qwen2.5-0.5B --task ag_news --exp order --k 8 --seeds 3 --perms 10

# GSM8K: CoT vs. direct vs. corrupted demos
python scripts/run_experiment.py --model Qwen/Qwen2.5-1.5B --task gsm8k --exp cot --k 4 --n-test 300

# Any OpenAI-compatible API (OpenAI, `vllm serve`, Together, DeepSeek, ...)
python scripts/run_experiment.py --backend api --api-mode chat --model gpt-4o-mini --task gsm8k --exp shots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from icl.models import load_model  # noqa: E402
from icl.runner import CLS_EXPERIMENTS, GEN_EXPERIMENTS, Runner, build_conditions  # noqa: E402
from icl.tasks import ALL_TASKS, load_task  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF model id / local path, or API model name")
    p.add_argument("--task", nargs="+", required=True, choices=ALL_TASKS)
    p.add_argument("--exp", required=True, choices=sorted(set(CLS_EXPERIMENTS + GEN_EXPERIMENTS)))
    p.add_argument("--shots", type=int, nargs="+", default=None,
                   help="k values for --exp shots (default: 0 1 2 4 8 16 32 for classification, 0 1 2 4 8 for gsm8k)")
    p.add_argument("--k", type=int, default=None, help="number of demos for labels/order/template/cot (default 16 cls, 4 gen)")
    p.add_argument("--seeds", type=int, default=5, help="number of independently sampled demo sets")
    p.add_argument("--perms", type=int, default=10, help="permutations per demo set (--exp order)")
    p.add_argument("--n-test", type=int, default=500, help="test examples (fixed subsample, same for all conditions)")
    p.add_argument("--no-balance", action="store_true", help="sample demos uniformly instead of class-balanced")
    p.add_argument("--max-new-tokens", type=int, default=256, help="generation budget for gsm8k")
    p.add_argument("--out", default="results")
    p.add_argument("--no-save-predictions", action="store_true")
    # backend
    p.add_argument("--backend", choices=["hf", "api"], default="hf")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="auto", help="auto | bfloat16 | float16 | float32")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--base-url", default=None, help="API base URL (default: $OPENAI_BASE_URL)")
    p.add_argument("--api-mode", choices=["chat", "completion"], default="chat")
    p.add_argument("--max-workers", type=int, default=8)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    model = load_model(a.backend, a.model, batch_size=a.batch_size, device=a.device, dtype=a.dtype,
                       trust_remote_code=a.trust_remote_code, base_url=a.base_url, mode=a.api_mode,
                       max_workers=a.max_workers)
    for name in a.task:
        task = load_task(name, n_test=a.n_test)
        is_gen = task.type == "generation"
        shots = a.shots or ([0, 1, 2, 4, 8] if is_gen else [0, 1, 2, 4, 8, 16, 32])
        k = a.k if a.k is not None else (4 if is_gen else 16)
        conds = build_conditions(a.exp, task, shots=shots, k=k, seeds=a.seeds, perms=a.perms)
        print(f"== {name}: {len(task.test)} test examples, {len(conds)} conditions")
        runner = Runner(model, a.model, task, a.out, balanced=not a.no_balance,
                        max_new_tokens=a.max_new_tokens, save_predictions=not a.no_save_predictions)
        runner.run(conds)
        print(f"   -> {runner.out_dir(a.exp) / 'summary.csv'}")


if __name__ == "__main__":
    main()
