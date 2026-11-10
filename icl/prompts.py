"""Demonstration sampling, manipulation and prompt construction."""

from __future__ import annotations

import random
from collections import defaultdict

from .tasks import ABSTRACT_LABELS, ClassificationTask, GenerationTask, Template

# ----------------------------------------------------------------------------
# Demonstration sampling
# ----------------------------------------------------------------------------


def sample_demos(pool: list[dict], k: int, seed: int, balanced: bool = True) -> list[dict]:
    """Sample k demonstrations. If ``balanced`` and the pool has labels, sample
    round-robin over classes so every label appears as evenly as possible."""
    if k == 0:
        return []
    rng = random.Random(seed)
    if not balanced or "label" not in pool[0]:
        return rng.sample(pool, k)
    by_label = defaultdict(list)
    for ex in pool:
        by_label[ex["label"]].append(ex)
    labels = sorted(by_label)
    rng.shuffle(labels)
    # how many per class
    counts = {l: k // len(labels) for l in labels}
    for l in labels[: k % len(labels)]:
        counts[l] += 1
    demos = []
    for l in labels:
        demos += rng.sample(by_label[l], counts[l])
    rng.shuffle(demos)
    return demos


def permute(demos: list, perm_seed: int | None) -> list:
    """Return the demos in a different order (None = keep sampled order)."""
    if perm_seed is None:
        return list(demos)
    out = list(demos)
    random.Random(10_000 + perm_seed).shuffle(out)
    return out


# ----------------------------------------------------------------------------
# Label manipulation (classification)
# ----------------------------------------------------------------------------

LABEL_MODES = ["gold", "random", "flipped", "abstract"]


def demo_labels(demos: list[dict], mode: str, num_labels: int, seed: int) -> list[int]:
    """Labels shown in the demonstrations.

    gold     - true labels
    random   - uniformly random labels (Min et al. 2022)
    flipped  - systematic remapping y -> (y + 1) % C (Wei et al. 2023);
               for binary tasks this is exactly "flip every label"
    abstract - true labels, but rendered with abstract words (see verbalizers_for)
    """
    gold = [d["label"] for d in demos]
    if mode in ("gold", "abstract"):
        return gold
    if mode == "random":
        rng = random.Random(20_000 + seed)
        return [rng.randrange(num_labels) for _ in gold]
    if mode == "flipped":
        return [flip(y, num_labels) for y in gold]
    raise ValueError(f"Unknown label mode {mode!r}; choose from {LABEL_MODES}")


def flip(y: int, num_labels: int) -> int:
    return (y + 1) % num_labels


def verbalizers_for(template: Template, mode: str) -> list[str]:
    if mode == "abstract":
        return ABSTRACT_LABELS[: len(template.verbalizers)]
    return template.verbalizers


# ----------------------------------------------------------------------------
# Prompt building
# ----------------------------------------------------------------------------


def build_cls_prompt(
    template: Template,
    demos: list[dict],
    labels: list[int],
    verbalizers: list[str],
    query: dict,
) -> str:
    parts = [template.render_input(d) + template.render_label(verbalizers[y]) for d, y in zip(demos, labels)]
    parts.append(template.render_input(query))
    body = template.sep.join(parts)
    return (template.instruction + template.sep + body) if template.instruction else body


def cls_choices(verbalizers: list[str]) -> list[str]:
    return [Template.render_label(v) for v in verbalizers]


# ----------------------------------------------------------------------------
# Generation (GSM8K) demos
# ----------------------------------------------------------------------------

COT_MODES = ["cot", "direct", "wrong_answer", "shuffled_rationale"]


def gen_demo_strings(task: GenerationTask, demos: list[dict], mode: str, seed: int) -> list[str]:
    """Render math demonstrations under different conditions.

    cot                - gold step-by-step rationale + gold answer
    direct             - answer only (no reasoning)
    wrong_answer       - gold rationale, but the final answer is corrupted
    shuffled_rationale - rationales are shuffled across demos (reasoning no
                         longer matches the question), answers stay gold
    """
    rng = random.Random(30_000 + seed)
    rationales = [d["rationale"] for d in demos]
    if mode == "shuffled_rationale" and len(demos) > 1:
        # derangement-ish: rotate by one so no demo keeps its own rationale
        rationales = rationales[1:] + rationales[:1]
    out = []
    for d, r in zip(demos, rationales):
        ans = d["answer"]
        if mode == "wrong_answer":
            ans = _corrupt_number(ans, rng)
        rationale = None if mode == "direct" else r
        if mode not in COT_MODES:
            raise ValueError(f"Unknown CoT mode {mode!r}; choose from {COT_MODES}")
        out.append(task.format_question(d["question"]) + task.format_answer(rationale, ans))
    return out


def _corrupt_number(ans: str, rng: random.Random) -> str:
    try:
        v = float(ans)
    except ValueError:
        return ans + "1"
    delta = rng.choice([-1, 1]) * rng.randint(1, max(2, int(abs(v) * 0.5) + 1))
    w = v + delta
    return str(int(w)) if w == int(w) else f"{w:.2f}"


def build_gen_prompt(task: GenerationTask, demo_strings: list[str], question: str) -> str:
    return task.sep.join(demo_strings + [task.format_question(question)])
