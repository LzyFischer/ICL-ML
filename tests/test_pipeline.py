"""End-to-end smoke tests that run fully offline (no HF Hub access needed).

A tiny randomly-initialised GPT-2 and a locally trained BPE tokenizer stand in
for a real model, and the synthetic tasks stand in for the real datasets.
Accuracy is meaningless here; we test that every code path runs and that the
outputs have the right shape.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from icl import prompts as P  # noqa: E402
from icl.tasks import load_task  # noqa: E402


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    d = tmp_path_factory.mktemp("tiny")
    corpus = []
    for name in ["synthetic", "synthetic2"]:
        t = load_task(name, n_test=None)
        for ex in t.train[:300]:
            corpus.append(ex["text"])
    corpus += ["Input: Output: Sequence: Class: A B left right red blue foo bar Query Candidate Label Yes No N/A"]
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=600, special_tokens=["<eos>"],
                                                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<eos>")
    hf_tok.save_pretrained(d)
    cfg = GPT2Config(vocab_size=len(hf_tok), n_positions=2048, n_embd=32, n_layer=2, n_head=2,
                     eos_token_id=hf_tok.eos_token_id, bos_token_id=hf_tok.eos_token_id)
    GPT2LMHeadModel(cfg).save_pretrained(d)
    return str(d)


# ---------------------------------------------------------------- unit tests


def test_balanced_sampling():
    task = load_task("synthetic", n_test=10)
    demos = P.sample_demos(task.train, 8, seed=0)
    labels = [d["label"] for d in demos]
    assert len(demos) == 8 and labels.count(0) == labels.count(1) == 4
    assert P.sample_demos(task.train, 8, seed=0) == demos  # deterministic


def test_label_modes():
    demos = [{"label": y} for y in [0, 1, 2, 3]]
    assert P.demo_labels(demos, "flipped", 4, 0) == [1, 2, 3, 0]
    assert P.demo_labels(demos, "gold", 4, 0) == [0, 1, 2, 3]
    assert all(0 <= y < 4 for y in P.demo_labels(demos, "random", 4, 0))


def test_prompt_format():
    task = load_task("synthetic", n_test=5)
    tpl = task.templates[0]
    demos = P.sample_demos(task.train, 2, 0)
    prompt = P.build_cls_prompt(tpl, demos, [0, 1], tpl.verbalizers, task.test[0])
    assert prompt.count("Output:") == 3 and prompt.endswith("Output:")
    assert "Output: A\n\n" in prompt and "Output: B\n\n" in prompt


# ---------------------------------------------------------------- end to end


@pytest.mark.parametrize("exp", ["shots", "labels", "order", "template"])
def test_cls_experiments(tiny_model_dir, tmp_path, exp):
    from scripts.run_experiment import main

    main(["--model", tiny_model_dir, "--task", "synthetic", "--exp", exp, "--shots", "0", "2", "4",
          "--k", "4", "--seeds", "2", "--perms", "2", "--n-test", "20", "--out", str(tmp_path), "--device", "cpu"])
    df = pd.read_csv(next(tmp_path.glob(f"{exp}/synthetic/*/summary.csv")))
    assert (df.status == "ok").all() and df.acc.between(0, 1).all() and df.acc_cal.between(0, 1).all()
    if exp == "labels":
        assert set(df["mode"]) == set(P.LABEL_MODES) | {"gold"}
        assert df[df["mode"] == "flipped"].acc_flip.notna().all()


def test_multitoken_scoring_matches_single(tiny_model_dir):
    """The slow (per-pair) scorer and the fast (single-token) scorer agree."""
    from icl.models import HFModel

    m = HFModel(tiny_model_dir, device="cpu", batch_size=3)
    prompts = ["Input: dax wug\nOutput:", "Input: zib\nOutput:", "Sequence: mek fep\nClass:"]
    choices = [" A", " B"]
    fast = m.score_choices(prompts, choices)
    pair = m.score_choices(prompts, choices + [" A B"])[:, :2]  # 3rd choice forces the pair path
    np.testing.assert_allclose(fast, pair, atol=1e-4)


def test_resume(tiny_model_dir, tmp_path):
    from scripts.run_experiment import main

    args = ["--model", tiny_model_dir, "--task", "synthetic", "--exp", "shots", "--shots", "0", "2",
            "--seeds", "1", "--n-test", "10", "--out", str(tmp_path), "--device", "cpu"]
    main(args)
    main(args)  # second run must not duplicate rows
    df = pd.read_csv(next(tmp_path.glob("shots/synthetic/*/summary.csv")))
    assert len(df) == 2


def test_too_long_is_skipped(tiny_model_dir, tmp_path):
    from scripts.run_experiment import main

    main(["--model", tiny_model_dir, "--task", "synthetic", "--exp", "shots", "--shots", "512",
          "--seeds", "1", "--n-test", "4", "--out", str(tmp_path), "--device", "cpu"])
    df = pd.read_csv(next(tmp_path.glob("shots/synthetic/*/summary.csv")))
    assert (df.status == "too_long").all()


def test_plotting(tiny_model_dir, tmp_path):
    import subprocess

    from scripts.run_experiment import main

    res = tmp_path / "res"
    for exp in ["shots", "labels", "order", "template"]:
        main(["--model", tiny_model_dir, "--task", "synthetic", "--exp", exp, "--shots", "0", "2", "4",
              "--k", "4", "--seeds", "2", "--perms", "3", "--n-test", "16", "--out", str(res), "--device", "cpu"])
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / "scripts/plot_results.py"), "--results", str(res),
                    "--out", str(tmp_path / "fig")], check=True)
    pngs = {p.name for p in (tmp_path / "fig").glob("*.png")}
    assert {"shots_acc.png", "labels_acc.png", "order_acc.png", "template_acc.png"} <= pngs
