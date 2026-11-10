"""Answer extraction and scoring helpers."""

from __future__ import annotations

import re

import numpy as np

_NUM = r"-?\d[\d,]*(?:\.\d+)?"
_ANSWER_IS = re.compile(r"answer is[:\s]*\$?\s*(" + _NUM + ")", re.IGNORECASE)
_ANY_NUM = re.compile(_NUM)


def extract_number(text: str) -> str | None:
    """'The answer is 42.' -> '42'; otherwise the last number in the text."""
    m = _ANSWER_IS.search(text)
    if m:
        return _norm(m.group(1))
    nums = _ANY_NUM.findall(text)
    return _norm(nums[-1]) if nums else None


def _norm(s: str) -> str:
    s = s.replace(",", "").rstrip(".")
    try:
        v = float(s)
        return str(int(v)) if v == int(v) else str(v)
    except ValueError:
        return s


def numbers_equal(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(_norm(gold))) < 1e-6
    except ValueError:
        return pred == _norm(gold)


def match_label(text: str, verbalizers: list[str]) -> int:
    """Map a generated string to a label index (-1 if none matches).
    Used only by backends without log-prob scoring."""
    t = text.strip().strip(".\"'*").lower()
    for i, v in enumerate(verbalizers):  # exact / prefix match first
        if t == v.lower() or t.startswith(v.lower()):
            return i
    hits = [i for i, v in enumerate(verbalizers) if re.search(r"\b" + re.escape(v.lower()) + r"\b", t)]
    return hits[0] if len(hits) == 1 else -1


def calibrate(scores: np.ndarray, cf_scores: np.ndarray) -> np.ndarray:
    """Contextual calibration (Zhao et al. 2021): divide label probabilities by
    the probabilities assigned to a content-free input ("N/A")."""
    def norm_logp(s):
        s = s - s.max(axis=-1, keepdims=True)
        return s - np.log(np.exp(s).sum(axis=-1, keepdims=True))
    return norm_logp(scores) - norm_logp(cf_scores)


def macro_f1(y_true: list[int], y_pred: list[int], num_labels: int) -> float:
    f1s = []
    for c in range(num_labels):
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))
