"""Experiment definitions and the evaluation loop.

An *experiment* is a list of *conditions*. Each condition fixes the number of
shots, how demonstrations are manipulated, the template, and the seeds. All
conditions are evaluated on the same test examples, and one summary row per
condition is appended to ``summary.csv`` (so interrupted runs can resume).
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from . import prompts as P
from .metrics import calibrate, extract_number, macro_f1, match_label, numbers_equal
from .tasks import ClassificationTask, GenerationTask

CLS_EXPERIMENTS = ["shots", "labels", "order", "template"]
GEN_EXPERIMENTS = ["shots", "cot", "order"]


SUMMARY_FIELDS = [
    "model", "task", "exp", "k", "mode", "template", "demo_seed", "perm_seed", "status", "n",
    "acc", "acc_cal", "acc_bc", "macro_f1", "acc_flip", "majority_pred_frac", "invalid_frac",
    "llm_frac", "selector_prompts", "selector_tokens", "dropped_demos",
    "mean_output_chars", "mean_prompt_tokens", "seconds", "condition",
]


@dataclass(frozen=True)
class Condition:
    exp: str
    k: int
    mode: str = "gold"  # label mode (classification) or CoT mode (generation)
    template: int = 0
    demo_seed: int = 0
    perm_seed: int | None = None

    def key(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def build_conditions(exp: str, task, shots: list[int], k: int, seeds: int, perms: int) -> list[Condition]:
    is_gen = isinstance(task, GenerationTask)
    base_mode = "cot" if is_gen else "gold"
    valid = GEN_EXPERIMENTS if is_gen else CLS_EXPERIMENTS
    if exp not in valid:
        raise ValueError(f"Experiment {exp!r} not available for {task.name}; choose from {valid}")
    S = range(seeds)

    if exp == "shots":
        out = []
        for kk in shots:  # k=0 is deterministic: one seed is enough
            out += [Condition(exp, kk, base_mode, 0, s) for s in (range(1) if kk == 0 else S)]
        return out
    if exp == "labels":
        out = [Condition(exp, 0, "gold", 0, 0)]  # zero-shot reference
        return out + [Condition(exp, k, m, 0, s) for m in P.LABEL_MODES for s in S]
    if exp == "cot":
        out = [Condition(exp, 0, "cot", 0, 0)]
        return out + [Condition(exp, k, m, 0, s) for m in P.COT_MODES for s in S]
    if exp == "order":
        return [Condition(exp, k, base_mode, 0, s, p) for s in S for p in range(perms)]
    if exp == "template":
        return [Condition(exp, k, "gold", t, s) for t in range(len(task.templates)) for s in S]
    raise ValueError(exp)


def _slug(name: str) -> str:
    return name.rstrip("/").split("/")[-1].replace(":", "_")


class Runner:
    def __init__(self, model, model_name: str, task, out_root: str | Path, balanced: bool = True,
                 max_new_tokens: int = 256, save_predictions: bool = True):
        self.model = model
        self.model_name = model_name
        self.task = task
        self.out_root = Path(out_root)
        self.balanced = balanced
        self.max_new_tokens = max_new_tokens
        self.save_predictions = save_predictions

    def out_dir(self, exp: str) -> Path:
        d = self.out_root / exp / self.task.name / _slug(self.model_name)
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _done_keys(path: Path) -> set[str]:
        if not path.exists():
            return set()
        with path.open() as f:
            return {r["condition"] for r in csv.DictReader(f)}

    def run(self, conditions: list[Condition]) -> list[dict]:
        if not conditions:
            return []
        out_dir = self.out_dir(conditions[0].exp)
        summary_path = out_dir / "summary.csv"
        done = self._done_keys(summary_path)
        todo = [c for c in conditions if c.key() not in done]
        if len(todo) < len(conditions):
            print(f"[resume] {len(conditions) - len(todo)} conditions already done in {summary_path}")
        rows = []
        for cond in tqdm(todo, desc=f"{self.task.name}/{cond_exp(conditions)}/{_slug(self.model_name)}"):
            t0 = time.time()
            if isinstance(self.task, ClassificationTask):
                row, preds = self._eval_cls(cond)
            else:
                row, preds = self._eval_gen(cond)
            row = {"model": self.model_name, "task": self.task.name, **asdict(cond), **row,
                   "seconds": round(time.time() - t0, 1), "condition": cond.key()}
            self._append_csv(summary_path, row)
            if self.save_predictions and preds:
                with (out_dir / "predictions.jsonl").open("a") as f:
                    for p in preds:
                        f.write(json.dumps({"condition": cond.key(), **p}) + "\n")
            rows.append(row)
            tqdm.write(_fmt(row))
        return rows

    @staticmethod
    def _append_csv(path: Path, row: dict):
        new = not path.exists()
        with path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, restval="", extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)

    def _too_long(self, prompts: list[str], reserve: int) -> tuple[bool, float]:
        lens = [self.model.n_tokens(p) for p in prompts]
        return max(lens) + reserve > self.model.max_len, float(np.mean(lens))

    # -- classification ------------------------------------------------------

    def _eval_cls(self, c: Condition):
        task: ClassificationTask = self.task
        tpl = task.templates[c.template]
        C = task.num_labels
        demos = P.permute(P.sample_demos(task.train, c.k, c.demo_seed, self.balanced), c.perm_seed)
        shown = P.demo_labels(demos, c.mode, C, c.demo_seed)
        verb = P.verbalizers_for(tpl, c.mode)
        choices = P.cls_choices(verb)
        prompts = [P.build_cls_prompt(tpl, demos, shown, verb, x) for x in task.test]
        gold = [x["label"] for x in task.test]

        too_long, mean_len = self._too_long(prompts, reserve=8)
        if too_long:
            return {"status": "too_long", "mean_prompt_tokens": round(mean_len)}, []

        if self.model.supports_scoring:
            scores = self.model.score_choices(prompts, choices)
            pred = scores.argmax(1).tolist()
            cf_prompt = P.build_cls_prompt(tpl, demos, shown, verb, task.content_free())
            cf = self.model.score_choices([cf_prompt], choices)
            pred_cal = calibrate(scores, cf).argmax(1).tolist()
        else:
            gens = self.model.generate(prompts, max_new_tokens=5, stop=["\n"])
            pred = [match_label(g, verb) for g in gens]
            pred_cal, scores = None, None

        row = {
            "status": "ok",
            "n": len(gold),
            "acc": _mean([p == y for p, y in zip(pred, gold)]),
            "macro_f1": macro_f1(gold, pred, C),
            "acc_cal": _mean([p == y for p, y in zip(pred_cal, gold)]) if pred_cal else float("nan"),
            # how often the model follows the (corrupted) mapping shown in the demos
            "acc_flip": _mean([p == P.flip(y, C) for p, y in zip(pred, gold)]) if c.mode == "flipped" else float("nan"),
            "majority_pred_frac": max(pred.count(i) for i in range(C)) / len(pred),
            "invalid_frac": pred.count(-1) / len(pred),
            "mean_prompt_tokens": round(mean_len),
        }
        preds = [{"idx": i, "gold": y, "pred": p, **({"pred_cal": pred_cal[i]} if pred_cal else {}),
                  **({"scores": [round(float(s), 4) for s in scores[i]]} if scores is not None else {})}
                 for i, (y, p) in enumerate(zip(gold, pred))]
        return row, preds

    # -- generation ------------------------------------------------------------

    def _eval_gen(self, c: Condition):
        task: GenerationTask = self.task
        demos = P.permute(P.sample_demos(task.train, c.k, c.demo_seed, balanced=False), c.perm_seed)
        demo_str = P.gen_demo_strings(task, demos, c.mode, c.demo_seed)
        prompts = [P.build_gen_prompt(task, demo_str, x["question"]) for x in task.test]
        too_long, mean_len = self._too_long(prompts, reserve=self.max_new_tokens)
        if too_long:
            return {"status": "too_long", "mean_prompt_tokens": round(mean_len)}, []
        gens = self.model.generate(prompts, max_new_tokens=self.max_new_tokens, stop=task.stop)
        extracted = [extract_number(g) for g in gens]
        correct = [numbers_equal(e, x["answer"]) for e, x in zip(extracted, task.test)]
        row = {
            "status": "ok",
            "n": len(gens),
            "acc": _mean(correct),
            "invalid_frac": _mean([e is None for e in extracted]),
            "mean_output_chars": round(float(np.mean([len(g) for g in gens])), 1),
            "mean_prompt_tokens": round(mean_len),
        }
        preds = [{"idx": i, "gold": x["answer"], "pred": e, "correct": ok, "output": g}
                 for i, (x, g, e, ok) in enumerate(zip(task.test, gens, extracted, correct))]
        return row, preds


def cond_exp(conds: list[Condition]) -> str:
    return conds[0].exp if conds else ""


def _mean(xs) -> float:
    xs = list(xs)
    return float(np.mean(xs)) if xs else float("nan")


def _fmt(row: dict) -> str:
    keys = ["k", "mode", "template", "demo_seed", "perm_seed", "status", "acc", "acc_cal", "acc_flip"]
    parts = []
    for k in keys:
        v = row.get(k)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        parts.append(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}")
    return "  " + " ".join(parts)


# ----------------------------------------------------------------------------
# Method comparison: per-query demonstration selection
# ----------------------------------------------------------------------------


def run_method(model, model_name: str, task, selectors: dict, k: int, orders: list[int], out_root,
               seed: int = 0, save_predictions: bool = True) -> list[dict]:
    """Evaluate each selector on the test set. Selection is done once per
    selector; the chosen demos are then evaluated under every order seed
    (order 0 = main results, all orders = robustness to ordering)."""
    from .selection import evaluate_demos

    out_dir = Path(out_root) / "method" / task.name / _slug(model_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = out_dir / "summary.csv"
    done = Runner._done_keys(summary)
    rows = []
    for name, selector in selectors.items():
        conds = [Condition("method", k, name, 0, seed, o) for o in orders]
        todo = [c for c in conds if c.key() not in done]
        if not todo:
            print(f"[resume] {task.name}/{name}: done")
            continue
        t0 = time.time()
        sel = selector.select(task.test, k, seed=seed)
        t_sel = time.time() - t0
        for c in todo:
            t1 = time.time()
            ev = evaluate_demos(model, task, task.test, sel.demos, order_seed=c.perm_seed)
            row = {"model": model_name, "task": task.name, **asdict(c), "status": "ok", "n": len(task.test),
                   **{f: ev[f] for f in ["acc", "acc_bc", "macro_f1", "majority_pred_frac", "invalid_frac",
                                         "dropped_demos"]},
                   "mean_prompt_tokens": round(ev["mean_prompt_tokens"]),
                   "llm_frac": float(np.mean(sel.llm_used)) if sel.llm_used else 0.0,
                   "selector_prompts": sel.selector_prompts, "selector_tokens": sel.selector_tokens,
                   "seconds": round(time.time() - t1 + (t_sel if c is todo[0] else 0), 1), "condition": c.key()}
            Runner._append_csv(summary, row)
            if save_predictions:
                with (out_dir / "predictions.jsonl").open("a") as f:
                    for i, (y, p) in enumerate(zip(ev["gold"], ev["pred"])):
                        f.write(json.dumps({"condition": c.key(), "idx": i, "gold": y, "pred": p,
                                            "demos": [int(j) for j in sel.demos[i]]}) + "\n")
            rows.append(row)
            print(f"  {task.name} {name:<22} order={c.perm_seed} acc={row['acc']:.3f} acc_bc={row['acc_bc']:.3f}"
                  f" llm_frac={row['llm_frac']:.2f}")
    return rows
