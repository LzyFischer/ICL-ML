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
from icl.runner import CLS_EXPERIMENTS, GEN_EXPERIMENTS, Runner, build_conditions, run_method  # noqa: E402
from icl.tasks import ALL_TASKS, load_task  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF model id / local path, or API model name")
    p.add_argument("--task", nargs="+", required=True, choices=ALL_TASKS)
    p.add_argument("--exp", required=True, choices=sorted(set(CLS_EXPERIMENTS + GEN_EXPERIMENTS + ["method"])))
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
    # --exp method (per-query demonstration selection)
    p.add_argument("--methods", nargs="+",
                   default=["random", "knn", "topm_random", "llm_norules", "llm_human", "llm_learned",
                            "llm_learned_adaptive"],
                   help="selectors to compare; llm_* need a HF selector model, llm_learned needs --rules")
    p.add_argument("--selector-model", default=None, help="selector LLM (default: same as --model)")
    p.add_argument("--rules", default=None, help="learned rulebook file (best_rules.txt)")
    p.add_argument("--top-m", type=int, default=20, help="retrieved candidates per query")
    p.add_argument("--alpha", type=float, default=1.0, help="weight of retrieval similarity in LLM selectors")
    p.add_argument("--tau", type=float, default=1.0, help="adaptive gate: skip the LLM if kNN labels agree >= tau")
    p.add_argument("--orders", type=int, default=1, help="order seeds to evaluate (0 = main results)")
    p.add_argument("--pool-size", type=int, default=5000, help="cap on the demonstration pool")
    p.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2", help="or 'hash' (offline)")
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
    if a.exp == "method":
        return main_method(a, model)
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


def main_method(a, model):
    from icl.retrieval import Retriever, load_embedder
    from icl.selection import make_selector

    methods = list(a.methods)
    rules = Path(a.rules).read_text() if a.rules else None
    if rules is None:
        dropped = [m for m in methods if m.startswith("llm_learned")]
        if dropped:
            print(f"[info] no --rules given; skipping {dropped}")
        methods = [m for m in methods if not m.startswith("llm_learned")]
    scorer = None
    if any(m.startswith("llm") for m in methods):
        if a.selector_model and a.selector_model != a.model:
            scorer = load_model("hf", a.selector_model, batch_size=a.batch_size, device=a.device, dtype=a.dtype)
        elif getattr(model, "supports_scoring", False):
            scorer = model
        else:
            raise SystemExit("LLM selectors need a HF selector model: pass --selector-model")
    embedder = load_embedder(a.embedder, a.device)
    for name in a.task:
        task = load_task(name, n_test=a.n_test, pool_size=a.pool_size)
        if task.type != "classification":
            raise SystemExit("--exp method currently supports classification tasks only")
        retriever = Retriever(embedder, task.train, task.text_fields, cache_dir=Path(a.out) / "emb_cache")
        selectors = {m: make_selector(m, task, retriever, scorer, rules, a.top_m, a.alpha, a.tau) for m in methods}
        print(f"== {name}: {len(task.test)} test, pool {len(task.train)}, methods {methods}")
        run_method(model, a.model, task, selectors, k=a.k if a.k is not None else 8, orders=list(range(a.orders)),
                   out_root=a.out, save_predictions=not a.no_save_predictions)


if __name__ == "__main__":
    main()
