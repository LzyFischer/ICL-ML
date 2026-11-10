#!/usr/bin/env python
"""Run one experiment for one model on one or more tasks.

Analysis experiments (fixed demos, class-balanced random sampling):
  shots     accuracy vs. number of demonstrations
  labels    gold / random / flipped / abstract demo labels (+ zero-shot)
  order     several demo sets x several orderings
  template  three prompt templates

Method comparison (per-query demo selection, see icl/selection.py):
  method    random / knn / topm_random / llm_norules / llm_human / llm_learned(_adaptive)

Examples
--------
python scripts/run_experiment.py --model Qwen/Qwen3.5-2B --task sst2 trec --exp shots
python scripts/run_experiment.py --model Qwen/Qwen3.5-2B --task sst2 --exp labels --k 16
python scripts/run_experiment.py --model Qwen/Qwen3.5-2B --task cr subj emotion --exp method \
    --rules rule_opt/qwen2b/best_rules.txt --orders 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from icl.models import HFModel  # noqa: E402
from icl.runner import EXPERIMENTS, Runner, build_conditions, run_method  # noqa: E402
from icl.tasks import ALL_TASKS, load_task  # noqa: E402

ALL_METHODS = ["random", "knn", "topm_random", "llm_norules", "llm_human", "llm_learned", "llm_learned_adaptive"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF model id or local path")
    p.add_argument("--task", nargs="+", required=True, choices=ALL_TASKS)
    p.add_argument("--exp", required=True, choices=EXPERIMENTS)
    p.add_argument("--k", type=int, default=8, help="number of demonstrations")
    p.add_argument("--n-test", type=int, default=500, help="test examples (fixed subsample)")
    p.add_argument("--out", default="results")
    # analysis experiments
    p.add_argument("--shots", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16, 32], help="k values for --exp shots")
    p.add_argument("--seeds", type=int, default=5, help="independently sampled demo sets")
    p.add_argument("--perms", type=int, default=10, help="orderings per demo set (--exp order)")
    # method comparison
    p.add_argument("--methods", nargs="+", default=ALL_METHODS)
    p.add_argument("--rules", default=None, help="learned rulebook (best_rules.txt); needed for llm_learned*")
    p.add_argument("--selector-model", default=None, help="selector LLM (default: same as --model)")
    p.add_argument("--orders", type=int, default=1, help="demo orders to evaluate (order 0 = main results)")
    p.add_argument("--top-m", type=int, default=20, help="retrieved candidates per query")
    p.add_argument("--pool-size", type=int, default=5000, help="cap on the demonstration pool")
    p.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2", help="or 'hash' (offline)")
    # hardware
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="auto", help="auto | bfloat16 | float16 | float32")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    model = HFModel(a.model, batch_size=a.batch_size, device=a.device, dtype=a.dtype)
    if a.exp == "method":
        return main_method(a, model)
    for name in a.task:
        task = load_task(name, n_test=a.n_test)
        conds = build_conditions(a.exp, task, shots=a.shots, k=a.k, seeds=a.seeds, perms=a.perms)
        print(f"== {name}: {len(task.test)} test examples, {len(conds)} conditions")
        runner = Runner(model, a.model, task, a.out)
        runner.run(conds)
        print(f"   -> {runner.out_dir(a.exp) / 'summary.csv'}")


def main_method(a, model):
    from icl.retrieval import Retriever, load_embedder
    from icl.selection import make_selector

    rules = Path(a.rules).read_text() if a.rules else None
    methods = [m for m in a.methods if rules or not m.startswith("llm_learned")]
    if len(methods) < len(a.methods):
        print("[info] no --rules given; skipping llm_learned*")
    scorer = model
    if a.selector_model and a.selector_model != a.model:
        scorer = HFModel(a.selector_model, batch_size=a.batch_size, device=a.device, dtype=a.dtype)
    embedder = load_embedder(a.embedder, a.device)
    for name in a.task:
        task = load_task(name, n_test=a.n_test, pool_size=a.pool_size)
        retriever = Retriever(embedder, task.train, task.text_fields, cache_dir=Path(a.out) / "emb_cache")
        selectors = {m: make_selector(m, task, retriever, scorer, rules, a.top_m) for m in methods}
        print(f"== {name}: {len(task.test)} test, pool {len(task.train)}, methods {methods}")
        run_method(model, a.model, task, selectors, k=a.k, orders=list(range(a.orders)), out_root=a.out)


if __name__ == "__main__":
    main()
