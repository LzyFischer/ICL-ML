"""Optimise the selector's *rules* with an LLM in the loop (OPRO-style).

What is optimised is the natural-language rulebook given to the selector LLM,
not any particular set of demonstrations. One round:

1. the selector (with the current rules) picks demos for every dev query of
   every meta-train task;
2. the target model is evaluated on those dev queries;
3. the optimizer LLM sees the rule history with scores plus diagnostics
   (accuracy vs. kNN, statistics of the chosen demos, failure cases) and
   writes a revised rulebook;
4. the new rulebook is evaluated; it becomes the current one if it scores
   higher (all attempts stay in the history the optimizer sees).

Score = mean over meta-train tasks of (dev accuracy with the rules - dev
accuracy of plain kNN), so the optimizer is pushed to beat the retrieval
baseline rather than to fit any one task. Dev queries are carved out of the
*training* split, never the test split; final evaluation is on unseen tasks.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .selection import KNNSelector, LLMSelector, evaluate_demos, selection_stats

OPTIMIZER_SYSTEM = (
    "You are an expert on in-context learning. You maintain a short rulebook that tells a selector model which "
    "demonstration examples to pick for a given query. A candidate's rule-compliance is judged one candidate at a "
    "time, so rules must be checkable from the query and a single candidate (with its label)."
)

OPTIMIZER_TEMPLATE = """We are searching for a rulebook that makes the selector choose better demonstrations than plain
similarity search (kNN). The rulebook must be GENERAL: it will be reused on new classification tasks with
different labels, so never mention specific tasks, topics or label names.

## Previous rulebooks and their scores (score = accuracy gain over kNN, averaged over tasks; higher is better)
{history}

## Diagnostics for the current rulebook (score {score:+.4f})
{diagnostics}

## Your job
Analyse why the current rulebook helps or hurts compared with kNN, then write an improved rulebook.
- at most {max_rules} rules, one line each, numbered;
- each rule must be checkable from the query and ONE candidate with its label;
- you may keep, delete, rewrite or add rules.

Think briefly, then output the new rulebook between <rules> and </rules>."""


def parse_rules(text: str, max_rules: int = 8) -> str | None:
    m = re.findall(r"<rules>(.*?)</rules>", text, flags=re.S | re.I)
    body = m[-1] if m else None
    if body is None:
        return None
    lines = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", ln).strip() for ln in body.strip().splitlines()]
    lines = [ln for ln in lines if len(ln) > 8][:max_rules]
    if not lines:
        return None
    return "\n".join(f"{i + 1}. {ln}" for i, ln in enumerate(lines))


def _short(s: str, n: int) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 3] + "..."


class RuleOptimizer:
    def __init__(self, target, scorer, optimizer, tasks, retrievers, k: int = 8, top_m: int = 20,
                 alpha: float = 1.0, max_rules: int = 8, n_cases: int = 2, out_dir: str | Path = "rule_opt",
                 optimizer_max_new_tokens: int = 1500, optimizer_thinking: bool = False):
        self.target, self.scorer, self.optimizer = target, scorer, optimizer
        self.tasks, self.retrievers = tasks, retrievers
        self.k, self.top_m, self.alpha = k, top_m, alpha
        self.max_rules, self.n_cases = max_rules, n_cases
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.opt_tokens, self.opt_thinking = optimizer_max_new_tokens, optimizer_thinking
        self.history: list[dict] = []
        self.knn = {}
        for t in tasks:  # the kNN reference is fixed: compute once
            demos = KNNSelector(t, retrievers[t.name]).select(t.dev, k).demos
            self.knn[t.name] = {"demos": demos, **evaluate_demos(target, t, t.dev, demos)}

    # -- evaluation ------------------------------------------------------------

    def evaluate(self, rules: str) -> dict:
        per_task = {}
        for t in self.tasks:
            sel = LLMSelector(t, self.retrievers[t.name], self.scorer, rules, self.top_m, self.alpha)
            demos = sel.select(t.dev, self.k).demos
            ev = evaluate_demos(self.target, t, t.dev, demos)
            per_task[t.name] = {"demos": demos, **ev, "stats": selection_stats(t, t.dev, demos)}
        score = float(np.mean([per_task[n]["acc"] - self.knn[n]["acc"] for n in per_task]))
        return {"score": score, "per_task": per_task}

    # -- feedback ------------------------------------------------------------------

    def diagnostics(self, res: dict) -> str:
        out = []
        for t in self.tasks:
            r, kb = res["per_task"][t.name], self.knn[t.name]
            verb = t.templates[0].verbalizers
            st, kst = r["stats"], selection_stats(t, t.dev, kb["demos"])
            pred_dist = Counter(verb[p] if p >= 0 else "?" for p in r["pred"])
            gold_dist = Counter(verb[g] for g in r["gold"])
            out.append(
                f"### Task {len(out) + 1} ({len(verb)} labels)\n"
                f"accuracy with rules {r['acc']:.3f} vs kNN {kb['acc']:.3f} (diff {r['acc'] - kb['acc']:+.3f})\n"
                f"chosen demos - rules vs kNN: same label as query {st['frac_demos_with_query_label']:.2f} vs "
                f"{kst['frac_demos_with_query_label']:.2f}; distinct labels per prompt "
                f"{st['distinct_labels_per_prompt']:.1f} vs {kst['distinct_labels_per_prompt']:.1f}; "
                f"words per demo {st['mean_demo_words']:.0f} vs {kst['mean_demo_words']:.0f}\n"
                f"target predictions {dict(pred_dist.most_common())} | gold {dict(gold_dist.most_common())}"
            )
            # failure cases (rules wrong, kNN right) and wins (rules right, kNN wrong)
            for kind, cond in [("LOSS (rules wrong, kNN right)", lambda a, b: not a and b),
                               ("WIN (rules right, kNN wrong)", lambda a, b: a and not b)]:
                idx = [i for i, (p, kp, g) in enumerate(zip(r["pred"], kb["pred"], r["gold"]))
                       if cond(p == g, kp == g)][: self.n_cases]
                for i in idx:
                    q = t.dev[i]
                    demos = "\n".join(
                        f"    [{verb[t.train[j]['label']]}] {_short(' | '.join(str(t.train[j][f]) for f in t.text_fields), 110)}"
                        for j in r["demos"][i])
                    out.append(f"{kind}: query \"{_short(' | '.join(str(q[f]) for f in t.text_fields), 200)}\" "
                               f"gold={verb[q['label']]} pred={verb[r['pred'][i]] if r['pred'][i] >= 0 else '?'}\n"
                               f"  demos chosen with rules:\n{demos}")
        return "\n".join(out)

    def history_text(self, last: int = 8) -> str:
        hs = sorted(self.history, key=lambda h: h["score"])[-last:]  # ascending, best last (OPRO)
        return "\n\n".join(f"score {h['score']:+.4f}:\n{h['rules']}" for h in hs)

    # -- main loop -----------------------------------------------------------------

    def run(self, init_rules: str, rounds: int = 10) -> str:
        cur = {"rules": init_rules, **self.evaluate(init_rules)}
        self._log(0, cur, accepted=True, reply=None)
        best = cur
        for rnd in range(1, rounds + 1):
            prompt = OPTIMIZER_TEMPLATE.format(history=self.history_text(), score=cur["score"],
                                               diagnostics=self.diagnostics(cur), max_rules=self.max_rules)
            reply = self.optimizer.chat(prompt, OPTIMIZER_SYSTEM, max_new_tokens=self.opt_tokens,
                                        thinking=self.opt_thinking)
            rules = parse_rules(reply, self.max_rules)
            if rules is None or any(h["rules"] == rules for h in self.history):
                print(f"[round {rnd}] optimizer produced no new rulebook; keeping current")
                self._log(rnd, {"rules": rules, "score": float("nan"), "per_task": {}}, accepted=False, reply=reply)
                continue
            cand = {"rules": rules, **self.evaluate(rules)}
            accepted = cand["score"] > cur["score"]
            self._log(rnd, cand, accepted, reply)
            if accepted:
                cur = cand
            if cand["score"] > best["score"]:
                best = cand
            print(f"[round {rnd}] score {cand['score']:+.4f} ({'accepted' if accepted else 'rejected'}); "
                  f"best {best['score']:+.4f}")
        (self.out_dir / "best_rules.txt").write_text(best["rules"] + "\n")
        return best["rules"]

    def _log(self, rnd: int, res: dict, accepted: bool, reply: str | None):
        if res.get("rules") is not None and not np.isnan(res["score"]):
            self.history.append({"rules": res["rules"], "score": res["score"]})
        entry = {
            "round": rnd, "accepted": accepted, "score": res["score"], "rules": res.get("rules"),
            "per_task": {n: {"acc": v["acc"], "acc_knn": self.knn[n]["acc"], "stats": v.get("stats")}
                         for n, v in res.get("per_task", {}).items()},
            "optimizer_reply": reply,
        }
        with (self.out_dir / "log.jsonl").open("a") as f:
            f.write(json.dumps(entry) + "\n")
