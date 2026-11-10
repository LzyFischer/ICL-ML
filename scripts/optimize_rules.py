#!/usr/bin/env python
"""Learn a selection rulebook on meta-train tasks.

python scripts/optimize_rules.py --target Qwen/Qwen3.5-2B --optimizer Qwen/Qwen3.5-9B \
    --tasks sst2 ag_news trec --rounds 10 --dev-size 200 --out rule_opt/qwen2b

The selector defaults to the target model ("the model picks its own demos").
The optimizer can be any HF chat model or, with --optimizer-backend api, any
OpenAI-compatible endpoint. Outputs <out>/best_rules.txt and <out>/log.jsonl.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from icl.models import load_model  # noqa: E402
from icl.retrieval import Retriever, load_embedder  # noqa: E402
from icl.rule_opt import RuleOptimizer  # noqa: E402
from icl.selection import HUMAN_RULES  # noqa: E402
from icl.tasks import load_task  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", required=True)
    p.add_argument("--selector", default=None, help="selector LLM (default: same as --target)")
    p.add_argument("--optimizer", default=None, help="optimizer LLM (default: same as --selector)")
    p.add_argument("--optimizer-backend", choices=["hf", "api"], default="hf")
    p.add_argument("--base-url", default=None)
    p.add_argument("--optimizer-thinking", action="store_true", help="let a hybrid-thinking optimizer think")
    p.add_argument("--tasks", nargs="+", default=["sst2", "ag_news", "trec"])
    p.add_argument("--dev-size", type=int, default=200)
    p.add_argument("--pool-size", type=int, default=5000)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--top-m", type=int, default=20)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--max-rules", type=int, default=8)
    p.add_argument("--init", choices=["human", "empty"], default="human")
    p.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2", help="or 'hash' (offline)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--out", default="rule_opt/run")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    kw = dict(batch_size=a.batch_size, device=a.device, dtype=a.dtype)
    target = load_model("hf", a.target, **kw)
    sel_name = a.selector or a.target
    scorer = target if sel_name == a.target else load_model("hf", sel_name, **kw)
    opt_name = a.optimizer or sel_name
    if a.optimizer_backend == "api":
        optimizer = load_model("api", opt_name, base_url=a.base_url)
    else:
        optimizer = {a.target: target, sel_name: scorer}.get(opt_name) or load_model("hf", opt_name, **kw)

    embedder = load_embedder(a.embedder, a.device)
    tasks, retrievers = [], {}
    for name in a.tasks:
        t = load_task(name, n_test=0, pool_size=a.pool_size, dev_size=a.dev_size)
        tasks.append(t)
        retrievers[name] = Retriever(embedder, t.train, t.text_fields, cache_dir=Path(a.out) / "emb_cache")
        print(f"{name}: pool {len(t.train)}, dev {len(t.dev)}")

    opt = RuleOptimizer(target, scorer, optimizer, tasks, retrievers, k=a.k, top_m=a.top_m, alpha=a.alpha,
                        max_rules=a.max_rules, out_dir=a.out, optimizer_thinking=a.optimizer_thinking)
    init = HUMAN_RULES if a.init == "human" else "1. Prefer examples that are relevant to the query."
    best = opt.run(init, rounds=a.rounds)
    print("\n=== best rules ===\n" + best)
    print(f"-> {Path(a.out) / 'best_rules.txt'}")


if __name__ == "__main__":
    main()
