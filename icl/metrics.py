"""Calibration and macro-F1."""

from __future__ import annotations

import numpy as np


def calibrate(scores: np.ndarray, cf_scores: np.ndarray) -> np.ndarray:
    """Contextual calibration (Zhao et al. 2021): divide label probabilities by
    the probabilities assigned to a content-free input ("N/A")."""
    def norm_logp(s):
        s = s - s.max(axis=-1, keepdims=True)
        return s - np.log(np.exp(s).sum(axis=-1, keepdims=True))
    return norm_logp(scores) - norm_logp(cf_scores)


def batch_calibrate(scores: np.ndarray) -> np.ndarray:
    """Batch calibration (Zhou et al. 2024): subtract the mean label
    probability over the evaluated batch, estimated without labels."""
    s = scores - scores.max(axis=1, keepdims=True)
    logp = s - np.log(np.exp(s).sum(axis=1, keepdims=True))
    return logp - np.log(np.exp(logp).mean(axis=0, keepdims=True))


def macro_f1(y_true: list[int], y_pred: list[int], num_labels: int) -> float:
    f1s = []
    for c in range(num_labels):
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))
