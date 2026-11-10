#!/usr/bin/env python
"""Aggregate every results/*/*/*/summary.csv into figures and a markdown table.

python scripts/plot_results.py --results results --out figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Fixed categorical order (assigned to models in sorted order, never cycled).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 11, "legend.frameon": False, "lines.linewidth": 2,
})

MODE_ORDER = {
    "labels": ["zero-shot", "gold", "random", "flipped", "abstract", "follows flipped"],
    "cot": ["zero-shot", "cot", "direct", "wrong_answer", "shuffled_rationale"],
}


def load(results: Path) -> pd.DataFrame:
    files = sorted(results.glob("*/*/*/summary.csv"))
    if not files:
        raise SystemExit(f"No summary.csv found under {results}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df["status"] == "ok"].copy()
    df["model_short"] = df["model"].astype(str).str.split("/").str[-1]
    return df


def colors_for(models) -> dict:
    return {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(sorted(models))}


def _axes(n):
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 3.6), squeeze=False)
    return fig, axes[0]


def _finish(fig, axes, path: Path, handles_from=0):
    h, l = axes[handles_from].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc="upper center", ncol=min(len(l), 4), bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def plot_shots(df, out, metric):
    tasks = sorted(df.task.unique())
    cmap = colors_for(df.model_short.unique())
    fig, axes = _axes(len(tasks))
    for ax, t in zip(axes, tasks):
        d = df[df.task == t]
        for m in sorted(d.model_short.unique()):
            g = d[d.model_short == m].groupby("k")[metric].agg(["mean", "std"]).reset_index().fillna(0)
            x = g["k"].to_numpy()
            ax.plot(x, g["mean"], marker="o", ms=5, color=cmap[m], label=m)
            ax.fill_between(x, g["mean"] - g["std"], g["mean"] + g["std"], color=cmap[m], alpha=0.15, lw=0)
        ax.set_xscale("symlog", base=2, linthresh=1)
        ks = sorted(d.k.unique())
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.set_title(t)
        ax.set_xlabel("# demonstrations (k)")
        ax.set_ylabel(metric)
    _finish(fig, axes, out / f"shots_{metric}.png")


def _grouped_bars(df, out, exp, metric, xcol, xorder, xlabel):
    tasks = sorted(df.task.unique())
    cmap = colors_for(df.model_short.unique())
    models = sorted(df.model_short.unique())
    fig, axes = _axes(len(tasks))
    for ax, t in zip(axes, tasks):
        d = df[df.task == t]
        cats = [c for c in xorder if c in set(d[xcol])]
        w = 0.8 / len(models)
        for i, m in enumerate(models):
            g = d[d.model_short == m].groupby(xcol)[metric].agg(["mean", "std"]).reindex(cats)
            xs = np.arange(len(cats)) + (i - (len(models) - 1) / 2) * w
            ax.bar(xs, g["mean"], width=w * 0.92, color=cmap[m], label=m,
                   yerr=g["std"].fillna(0), error_kw=dict(ecolor=INK2, lw=1, capsize=2))
        ax.set_xticks(np.arange(len(cats)), cats, rotation=20, ha="right")
        ax.set_title(t)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(metric)
        ax.grid(axis="x", visible=False)
    _finish(fig, axes, out / f"{exp}_{metric}.png")


def labels_frame(df):
    d = df.copy()
    d["cond"] = np.where(d.k == 0, "zero-shot", d["mode"])
    flip = d[d["mode"] == "flipped"].copy()
    if len(flip):  # "agreement with the flipped demo mapping" as its own bar
        flip["cond"] = "follows flipped"
        flip["acc"] = flip["acc_flip"]
        d = pd.concat([d, flip], ignore_index=True)
    return d


def plot_order(df, out, metric):
    tasks = sorted(df.task.unique())
    cmap = colors_for(df.model_short.unique())
    fig, axes = _axes(len(tasks))
    rng = np.random.default_rng(0)
    for ax, t in zip(axes, tasks):
        d = df[df.task == t]
        models = sorted(d.model_short.unique())
        for i, m in enumerate(models):
            dm = d[d.model_short == m]
            seeds = sorted(dm.demo_seed.unique())
            for j, s in enumerate(seeds):  # one column of points per demo set
                v = dm[dm.demo_seed == s][metric].to_numpy()
                x0 = i + (j - (len(seeds) - 1) / 2) * (0.7 / max(len(seeds), 1))
                ax.scatter(x0 + rng.uniform(-0.03, 0.03, len(v)), v, s=22, color=cmap[m], alpha=0.75,
                           edgecolor=SURFACE, linewidth=0.8, label=m if j == 0 else None)
                ax.hlines(v.mean(), x0 - 0.08, x0 + 0.08, color=INK, lw=1.5)
        ax.set_xticks(range(len(models)), models, rotation=20, ha="right")
        ax.set_title(f"{t}: each dot = one ordering")
        ax.set_ylabel(metric)
        ax.grid(axis="x", visible=False)
    _finish(fig, axes, out / f"order_{metric}.png")


def table(df: pd.DataFrame, out: Path):
    lines = []
    for exp, d in df.groupby("exp"):
        if exp == "labels":
            d = labels_frame(d)
            key = "cond"
        elif exp == "cot":
            d = d.assign(cond=np.where(d.k == 0, "zero-shot", d["mode"]))
            key = "cond"
        elif exp == "template":
            key = "template"
        elif exp == "order":
            key = "demo_seed"
        else:
            key = "k"
        g = d.groupby(["task", "model_short", key])["acc"].agg(["mean", "std", "min", "max", "count"]).reset_index()
        lines.append(f"## {exp}\n")
        lines.append(g.to_markdown(index=False, floatfmt=".3f") if _has_tabulate() else g.to_string(index=False))
        lines.append("")
    (out / "tables.md").write_text("\n".join(lines))
    print("wrote", out / "tables.md")


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="figures")
    p.add_argument("--metric", default="acc", help="acc | acc_cal | macro_f1")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load(Path(a.results))
    for exp, d in df.groupby("exp"):
        metric = a.metric if a.metric in d and d[a.metric].notna().any() else "acc"
        if exp == "shots":
            plot_shots(d, out, metric)
        elif exp == "labels":
            _grouped_bars(labels_frame(d), out, exp, "acc", "cond", MODE_ORDER["labels"], "demonstration labels")
        elif exp == "cot":
            d = d.assign(cond=np.where(d.k == 0, "zero-shot", d["mode"]))
            _grouped_bars(d, out, exp, "acc", "cond", MODE_ORDER["cot"], "demonstration format")
        elif exp == "order":
            plot_order(d, out, metric)
        elif exp == "template":
            d = d.assign(tpl=d["template"].astype(int).map(lambda t: f"template {t}"))
            _grouped_bars(d, out, exp, metric, "tpl", sorted(d.tpl.unique()), "prompt template")
    table(df, out)


if __name__ == "__main__":
    main()
