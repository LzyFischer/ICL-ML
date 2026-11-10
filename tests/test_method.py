"""Offline tests for per-query selection, rule optimisation and new loaders."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from icl import tasks as T  # noqa: E402
from icl.retrieval import HashEmbedder, Retriever  # noqa: E402
from icl.rule_opt import RuleOptimizer, parse_rules  # noqa: E402
from icl.selection import HUMAN_RULES, evaluate_demos, make_selector, selection_stats  # noqa: E402
from test_pipeline import tiny_model_dir  # noqa: E402,F401  (fixture)

CHAT_TEMPLATE = ("{% for m in messages %}<{{ m.role }}>{{ m.content }}\n{% endfor %}"
                 "{% if add_generation_prompt %}<assistant>{% endif %}")


@pytest.fixture(scope="module")
def model(tiny_model_dir):  # noqa: F811
    from icl.models import HFModel

    return HFModel(tiny_model_dir, device="cpu", batch_size=16)


@pytest.fixture(scope="module")
def task():
    return T.load_task("synthetic", n_test=12, pool_size=300, dev_size=10)


@pytest.fixture(scope="module")
def retriever(task):
    return Retriever(HashEmbedder(), task.train, task.text_fields)


def test_pool_dev_split(task):
    assert len(task.train) == 300 and len(task.dev) == 10 and len(task.test) == 12
    ids = {id(x) for x in task.train}
    assert not any(id(x) in ids for x in task.dev)


def test_retrieval_finds_exact_match(task, retriever):
    idx, sims = retriever.top_m([task.train[5]], 5)
    assert sims[0, 0] == pytest.approx(1.0, abs=1e-5)
    assert task.train[idx[0, 0]]["text"] == task.train[5]["text"]
    assert np.all(np.diff(sims[0]) <= 1e-6)  # sorted


@pytest.mark.parametrize("name", ["random", "knn", "topm_random", "llm_norules", "llm_human", "llm_human_adaptive"])
def test_selectors(task, retriever, model, name):
    sel = make_selector(name, task, retriever, model, None, top_m=10, alpha=1.0, tau=0.75)
    out = sel.select(task.test, 4, seed=0)
    assert len(out.demos) == len(task.test)
    for q, d in zip(task.test, out.demos):
        assert len(d) == 4 and len(set(d)) == 4
    if name != "random":  # retrieval-based methods choose from the top-M
        top, _ = retriever.top_m(task.test, 10)
        assert all(set(d) <= set(row) for d, row in zip(out.demos, top.tolist()))
    if name.endswith("adaptive"):
        assert 0 <= np.mean(out.llm_used) <= 1


def test_knn_equals_llm_with_large_alpha(task, retriever, model):
    knn = make_selector("knn", task, retriever).select(task.test, 3).demos
    llm = make_selector("llm_human", task, retriever, model, top_m=10, alpha=1e6).select(task.test, 3).demos
    # with alpha -> inf the fused score is the similarity; compare similarities (ties are common)
    sim = lambda q, d: sorted(float(retriever.emb[j] @ retriever.embedder.encode([q["text"]])[0]) for j in d)
    for q, a, b in zip(task.test, knn, llm):
        np.testing.assert_allclose(sim(q, a), sim(q, b), atol=1e-5)


def test_chat_prompt_paths(model):
    text, choices = model.chat_prompt("hi", "sys")
    assert choices == [" Yes", " No"] and text.endswith("Answer:")
    model.tok.chat_template = CHAT_TEMPLATE
    try:
        text, choices = model.chat_prompt("hi", "sys")
        assert choices == ["Yes", "No"] and text.endswith("<assistant>") and "<system>sys" in text
        assert isinstance(model.chat("hi", max_new_tokens=3), str)
    finally:
        model.tok.chat_template = None


def test_evaluate_demos_and_stats(task, retriever, model):
    demos = make_selector("knn", task, retriever).select(task.test, 4).demos
    ev0 = evaluate_demos(model, task, task.test, demos, order_seed=0)
    ev1 = evaluate_demos(model, task, task.test, demos, order_seed=1)
    assert 0 <= ev0["acc"] <= 1 and 0 <= ev0["acc_bc"] <= 1
    assert ev0["prompts"] != ev1["prompts"]  # order seed changes the order
    st = selection_stats(task, task.test, demos)
    assert 0 <= st["frac_demos_with_query_label"] <= 1


def test_parse_rules():
    txt = "thinking...\n<rules>\n1. Prefer similar examples.\n- Avoid very short examples.\nx\n</rules>"
    assert parse_rules(txt) == "1. Prefer similar examples.\n2. Avoid very short examples."
    assert parse_rules("no tags here") is None
    assert parse_rules("<rules>" + "\n".join(f"{i}. rule number {i}" for i in range(20)) + "</rules>",
                       max_rules=3).count("\n") == 2


class FakeOptimizer:
    """Returns a fixed sequence of replies (one per round)."""

    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def chat(self, user, system=None, max_new_tokens=0, thinking=False):
        self.prompts.append(user)
        return self.replies.pop(0)


def test_rule_optimizer(tmp_path, model):
    tasks = [T.load_task(n, n_test=0, pool_size=200, dev_size=8) for n in ["synthetic", "synthetic2"]]
    retr = {t.name: Retriever(HashEmbedder(), t.train, t.text_fields) for t in tasks}
    fake = FakeOptimizer(["<rules>1. Prefer examples with the same first word as the query.</rules>",
                          "garbage without tags",
                          "<rules>1. Prefer short examples.\n2. Prefer examples with a different label.</rules>"])
    opt = RuleOptimizer(model, model, fake, tasks, retr, k=3, top_m=6, out_dir=tmp_path)
    best = opt.run(HUMAN_RULES, rounds=3)
    log = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert len(log) == 4 and log[0]["round"] == 0 and log[2]["accepted"] is False
    assert (tmp_path / "best_rules.txt").read_text().strip() == best
    assert best in [HUMAN_RULES] + [h["rules"] for h in opt.history]
    # the optimizer saw diagnostics and the history
    assert "accuracy with rules" in fake.prompts[0] and "score" in fake.prompts[0]


def test_optimize_rules_script(tmp_path, tiny_model_dir):  # noqa: F811
    from scripts.optimize_rules import main

    main(["--target", tiny_model_dir, "--tasks", "synthetic", "synthetic2", "--dev-size", "6", "--pool-size", "100",
          "--k", "2", "--top-m", "4", "--rounds", "1", "--embedder", "hash", "--device", "cpu", "--out", str(tmp_path)])
    assert (tmp_path / "best_rules.txt").exists()


def test_method_experiment_and_plot(tmp_path, tiny_model_dir):  # noqa: F811
    import subprocess

    from scripts.run_experiment import main

    rules = tmp_path / "rules.txt"
    rules.write_text("1. Prefer examples similar to the query.\n")
    res = tmp_path / "res"
    main(["--model", tiny_model_dir, "--task", "synthetic", "--exp", "method", "--k", "3", "--top-m", "6",
          "--orders", "2", "--n-test", "10", "--pool-size", "150", "--embedder", "hash", "--rules", str(rules),
          "--out", str(res), "--device", "cpu"])
    df = pd.read_csv(next(res.glob("method/synthetic/*/summary.csv")))
    assert set(df["mode"]) == {"random", "knn", "topm_random", "llm_norules", "llm_human", "llm_learned",
                               "llm_learned_adaptive"}
    assert len(df) == 14 and set(df.perm_seed) == {0, 1}
    assert (df[df["mode"] == "knn"].selector_prompts == 0).all()
    assert (df[df["mode"] == "llm_human"].llm_frac == 1).all()
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / "scripts/plot_results.py"), "--results", str(res),
                    "--out", str(tmp_path / "fig")], check=True)
    assert (tmp_path / "fig" / "method_acc.png").exists()
    assert (tmp_path / "fig" / "method_orders_acc.png").exists()


def test_generic_loader_maps_label_names(monkeypatch):
    """A mirror whose label order differs from ours must be remapped by name."""
    from datasets import ClassLabel, Dataset, DatasetDict, Features, Value

    feats = Features({"text": Value("string"), "label": ClassLabel(names=["subjective", "objective"])})
    fake = DatasetDict({"train": Dataset.from_dict({"text": ["a b", "c"], "label": [0, 1]}, features=feats),
                        "test": Dataset.from_dict({"text": ["d"], "label": [1]}, features=feats)})
    calls = []

    def fake_hf(path, name=None):
        calls.append(path)
        if path == "broken/mirror":
            raise FileNotFoundError("nope")
        return fake

    monkeypatch.setattr(T, "_hf", fake_hf)
    tr, te, _ = T._load_text_cls([("broken/mirror", None, "text", "label"), ("ok", None, "text", "label")],
                                 {"objective": 0, "subjective": 1})
    assert calls == ["broken/mirror", "ok"]
    assert [r["label"] for r in tr] == [1, 0] and te[0]["label"] == 0
