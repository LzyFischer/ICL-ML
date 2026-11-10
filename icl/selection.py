"""Per-query demonstration selection (retrieve-then-select) and evaluation.

Stage 1: an embedding retriever returns the top-M most similar pool examples.
Stage 2: a selector picks k of them.

Selectors
---------
random        k random pool examples (no retrieval)
knn           the k most similar examples (Liu et al. 2022)
topm_random   k random examples from the top-M (controls for "better pool")
llm           pointwise LLM scoring: for every (query, candidate) pair the
              selector LLM is asked "Is this a good demonstration for the
              query, according to the rules?" and log P(Yes) - log P(No) is the
              score. Scores are z-normalised per query and fused with the
              retrieval similarity (weight ``alpha``). ``rules=None`` gives the
              "no rules" ablation.
llm + tau     adaptive: if the k nearest neighbours agree on a label with
              frequency >= tau, the query is "easy" and kNN is used directly;
              the LLM is only called for the remaining queries.

After selection all methods present demos in a random order (``order_seed``),
so differences come from *which* demos are chosen, not from ordering.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from . import prompts as P
from .metrics import batch_calibrate, macro_f1
from .retrieval import Retriever
from .tasks import ClassificationTask

HUMAN_RULES = """1. Prefer examples whose topic, wording or sentence structure is similar to the query.
2. Prefer examples whose label is clearly correct and unambiguous.
3. Prefer examples that are clear, natural and of moderate length.
4. Prefer examples from labels that the query could plausibly belong to, including confusable labels.
5. Avoid examples that are noisy, truncated, off-topic or that mainly contain names and numbers."""

SELECTOR_SYSTEM = (
    "You select demonstration examples for in-context learning. A small language model will be shown "
    "the selected examples (input and label) right before it has to label the query."
)


def selector_user_prompt(rules: str | None, query: str, candidate: str, label: str) -> str:
    parts = []
    if rules:
        parts.append(f"Rules for choosing a good demonstration:\n{rules.strip()}")
    parts.append(f"Query (its label is hidden):\n{query}")
    parts.append(f"Candidate demonstration:\n{candidate}\nLabel: {label}")
    if rules:
        parts.append("According to the rules, is this candidate a good demonstration for the query? Answer Yes or No.")
    else:
        parts.append("Is this candidate a helpful demonstration for the query? Answer Yes or No.")
    return "\n\n".join(parts)


@dataclass
class Selection:
    demos: list[list[int]]  # pool indices per query (most preferred first)
    llm_used: list[bool] = field(default_factory=list)
    selector_prompts: int = 0
    selector_tokens: int = 0


class BaseSelector:
    name = "base"

    def __init__(self, task: ClassificationTask, retriever: Retriever | None):
        self.task = task
        self.retriever = retriever

    def select(self, queries: list[dict], k: int, seed: int = 0) -> Selection:
        raise NotImplementedError


class RandomSelector(BaseSelector):
    name = "random"

    def select(self, queries, k, seed=0):
        rng = random.Random(seed)
        n = len(self.task.train)
        return Selection([rng.sample(range(n), k) for _ in queries])


class KNNSelector(BaseSelector):
    name = "knn"

    def select(self, queries, k, seed=0):
        idx, _ = self.retriever.top_m(queries, k)
        return Selection(idx.tolist())


class TopMRandomSelector(BaseSelector):
    name = "topm_random"

    def __init__(self, task, retriever, top_m: int = 20):
        super().__init__(task, retriever)
        self.top_m = top_m

    def select(self, queries, k, seed=0):
        rng = random.Random(seed)
        idx, _ = self.retriever.top_m(queries, self.top_m)
        return Selection([rng.sample(list(row), min(k, len(row))) for row in idx])


class LLMSelector(BaseSelector):
    def __init__(self, task, retriever, scorer, rules: str | None, top_m: int = 20, alpha: float = 1.0,
                 tau: float | None = None, name: str = "llm"):
        super().__init__(task, retriever)
        self.scorer = scorer
        self.rules = rules
        self.top_m = top_m
        self.alpha = alpha
        self.tau = tau
        self.name = name
        self.verbalizers = task.templates[0].verbalizers

    def _render(self, ex: dict) -> str:
        return " | ".join(str(ex[f]) for f in self.task.text_fields)

    def candidate_scores(self, queries: list[dict], cand_idx: np.ndarray) -> tuple[np.ndarray, int]:
        """LLM score log P(Yes) - log P(No) for every (query, candidate)."""
        prompts, choices = [], None
        for q, row in zip(queries, cand_idx):
            for j in row:
                c = self.task.train[j]
                user = selector_user_prompt(self.rules, self._render(q), self._render(c), self.verbalizers[c["label"]])
                text, choices = self.scorer.chat_prompt(user, SELECTOR_SYSTEM, answer_choices=("Yes", "No"))
                prompts.append(text)
        if not prompts:
            return np.zeros(cand_idx.shape), 0
        s = self.scorer.score_choices(prompts, list(choices), add_special_tokens=not self.scorer.has_chat_template)
        tokens = sum(self.scorer.n_tokens(p) for p in prompts[: min(len(prompts), 50)]) * len(prompts) // min(len(prompts), 50)
        return (s[:, 0] - s[:, 1]).reshape(cand_idx.shape), tokens

    def select(self, queries, k, seed=0):
        cand, sims = self.retriever.top_m(queries, self.top_m)
        # adaptive: queries whose k nearest neighbours agree on the label skip the LLM
        use_llm = np.ones(len(queries), dtype=bool)
        if self.tau is not None:
            for i, row in enumerate(cand[:, :k]):
                labels = [self.task.train[j]["label"] for j in row]
                use_llm[i] = Counter(labels).most_common(1)[0][1] / len(labels) < self.tau
        demos = [list(row[:k]) for row in cand]  # kNN default
        sel = np.where(use_llm)[0]
        tokens = 0
        if len(sel):
            llm, tokens = self.candidate_scores([queries[i] for i in sel], cand[sel])
            fused = _z(llm) + self.alpha * _z(sims[sel])
            for r, i in enumerate(sel):
                demos[i] = [int(cand[i, j]) for j in np.argsort(-fused[r])[:k]]
        return Selection(demos, use_llm.tolist(), int(use_llm.sum()) * cand.shape[1], tokens)


def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)


def make_selector(name: str, task, retriever, scorer=None, rules: str | None = None, top_m: int = 20,
                  alpha: float = 1.0, tau: float | None = None):
    """Factory. LLM selectors are named ``llm_norules`` / ``llm_human`` /
    ``llm_learned`` (+ ``_adaptive`` suffix to enable the tau gate)."""
    if name == "random":
        return RandomSelector(task, retriever)
    if name == "knn":
        return KNNSelector(task, retriever)
    if name == "topm_random":
        return TopMRandomSelector(task, retriever, top_m)
    if name.startswith("llm"):
        base = name.removesuffix("_adaptive")
        r = {"llm_norules": None, "llm_human": HUMAN_RULES}.get(base, rules)
        if base == "llm_learned" and not rules:
            raise ValueError("llm_learned needs --rules (output of scripts/optimize_rules.py)")
        return LLMSelector(task, retriever, scorer, r, top_m, alpha,
                           tau=(tau if tau is not None else 1.0) if name.endswith("_adaptive") else None, name=name)
    raise ValueError(f"Unknown selector {name!r}")


# ----------------------------------------------------------------------------
# Evaluate per-query demonstrations with the target model
# ----------------------------------------------------------------------------


def order_demos(idx: list[int], order_seed: int, qi: int) -> list[int]:
    out = list(idx)
    random.Random(order_seed * 1_000_003 + qi).shuffle(out)
    return out


def evaluate_demos(model, task: ClassificationTask, queries: list[dict], demo_idx: list[list[int]],
                   order_seed: int = 0, template: int = 0) -> dict:
    tpl = task.templates[template]
    verb = tpl.verbalizers
    prompts, n_dropped = [], 0
    for qi, (q, idx) in enumerate(zip(queries, demo_idx)):
        demos = [task.train[j] for j in order_demos(idx, order_seed, qi)]
        p = P.build_cls_prompt(tpl, demos, [d["label"] for d in demos], verb, q)
        while demos and model.n_tokens(p) + 8 > model.max_len:  # drop demos until it fits
            demos, n_dropped = demos[1:], n_dropped + 1
            p = P.build_cls_prompt(tpl, demos, [d["label"] for d in demos], verb, q)
        prompts.append(p)
    gold = [q["label"] for q in queries]
    C = len(verb)
    scores = model.score_choices(prompts, P.cls_choices(verb))
    pred = scores.argmax(1).tolist()
    pred_bc = batch_calibrate(scores).argmax(1).tolist()
    acc = float(np.mean([p == y for p, y in zip(pred, gold)]))
    return {
        "pred": pred, "gold": gold, "scores": scores, "prompts": prompts,
        "acc": acc,
        "acc_bc": float(np.mean([p == y for p, y in zip(pred_bc, gold)])),
        "macro_f1": macro_f1(gold, pred, C),
        "majority_pred_frac": max(pred.count(i) for i in range(C)) / len(pred),
        "dropped_demos": n_dropped,
        "mean_prompt_tokens": float(np.mean([model.n_tokens(p) for p in prompts[:100]])),
    }


def selection_stats(task: ClassificationTask, queries: list[dict], demo_idx: list[list[int]]) -> dict:
    """Cheap descriptive statistics of a selection (used as optimizer feedback)."""
    same = [np.mean([task.train[j]["label"] == q["label"] for j in idx]) for q, idx in zip(queries, demo_idx)]
    n_labels = [len({task.train[j]["label"] for j in idx}) for idx in demo_idx]
    length = [np.mean([len(" ".join(str(task.train[j][f]) for f in task.text_fields).split()) for j in idx])
              for idx in demo_idx]
    return {"frac_demos_with_query_label": float(np.mean(same)),
            "distinct_labels_per_prompt": float(np.mean(n_labels)),
            "mean_demo_words": float(np.mean(length))}
