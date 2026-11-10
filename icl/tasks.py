"""Tasks and dataset loading.

Only short-text classification tasks with few labels, so everything runs fast:

    sst2, cr, subj   (2 labels)     trec, emotion   (6 labels)     ag_news   (4 labels)

Every example is ``{"text": str, "label": int}``. Two offline synthetic tasks
(``synthetic``, ``synthetic2``) are used by the tests and need no download.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# Label words for the "abstract labels" condition (Min et al. 2022): semantically
# empty, so the model can only use them if it learns the mapping in context.
ABSTRACT_LABELS = ["foo", "bar", "baz", "qux", "zap", "wug"]


@dataclass
class Template:
    """``input_fmt`` ends right before the label; the label is appended as " word"."""

    input_fmt: str
    verbalizers: list[str]
    sep: str = "\n\n"

    def render_input(self, ex: dict) -> str:
        return self.input_fmt.format(**ex)

    @staticmethod
    def render_label(word: str) -> str:
        return " " + word


@dataclass
class ClassificationTask:
    name: str
    templates: list[Template]
    train: list[dict]  # demonstration pool
    test: list[dict]
    dev: list[dict] = field(default_factory=list)  # held out from train (rule optimisation only)
    text_fields: list[str] = field(default_factory=lambda: ["text"])

    @property
    def num_labels(self) -> int:
        return len(self.templates[0].verbalizers)

    def content_free(self) -> dict:
        """Content-free input for contextual calibration (Zhao et al. 2021)."""
        return {"text": "N/A"}


# --------------------------------------------------------------------------
# Templates (3 per task; template 0 is the default)
# --------------------------------------------------------------------------

TEMPLATES: dict[str, list[Template]] = {
    "sst2": [
        Template("Review: {text}\nSentiment:", ["negative", "positive"]),
        Template("{text} It was", ["terrible", "great"]),
        Template("Input: {text}\nOutput:", ["negative", "positive"]),
    ],
    "cr": [
        Template("Review: {text}\nSentiment:", ["negative", "positive"]),
        Template("{text} It was", ["terrible", "great"]),
        Template("Input: {text}\nOutput:", ["negative", "positive"]),
    ],
    "subj": [
        Template("Input: {text}\nType:", ["objective", "subjective"]),
        Template("{text} This is", ["factual", "opinion"]),
        Template("Sentence: {text}\nOutput:", ["objective", "subjective"]),
    ],
    "trec": [
        Template("Question: {text}\nAnswer type:", ["abbreviation", "entity", "description", "person", "location", "number"]),
        Template("{text}\nThis question asks for a", ["abbreviation", "thing", "description", "person", "place", "number"]),
        Template("Input: {text}\nOutput:", ["abbreviation", "entity", "description", "person", "location", "number"]),
    ],
    "emotion": [
        Template("Text: {text}\nEmotion:", ["sadness", "joy", "love", "anger", "fear", "surprise"]),
        Template("{text}\nThe writer feels", ["sad", "happy", "loving", "angry", "afraid", "surprised"]),
        Template("Input: {text}\nOutput:", ["sadness", "joy", "love", "anger", "fear", "surprise"]),
    ],
    "ag_news": [
        Template("Article: {text}\nTopic:", ["World", "Sports", "Business", "Technology"]),
        Template("{text}\nThis news is about", ["world", "sports", "business", "science"]),
        Template("Input: {text}\nOutput:", ["World", "Sports", "Business", "Technology"]),
    ],
}
for _name in ["synthetic", "synthetic2"]:
    TEMPLATES[_name] = [
        Template("Input: {text}\nOutput:", ["A", "B"]),
        Template("{text} =>", ["left", "right"]),
        Template("Sequence: {text}\nClass:", ["red", "blue"]),
    ]


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------


def _hf(path: str, name: str | None = None):
    from datasets import load_dataset

    return load_dataset(path, name)


def _norm_name(n) -> str:
    return str(n).strip().lower().replace("_", " ")


def _load_text_cls(candidates, aliases: dict[str, int]):
    """Try several HF mirrors in order. Each candidate is
    (path, config, text_col, label_col). Labels are mapped to our canonical
    order through their *names* (``aliases``: lowercased name -> index), so a
    mirror with a different label order cannot scramble the label words."""
    errors = []
    for path, cfg, text_col, label_col in candidates:
        try:
            ds = _hf(path, cfg)
        except Exception as e:  # noqa: BLE001 - try the next mirror
            errors.append(f"{path}: {type(e).__name__}: {e}")
            continue
        test_split = "test" if "test" in ds else "validation"
        feat = ds["train"].features.get(label_col)
        names = list(feat.names) if hasattr(feat, "names") else None
        if names is None and "label_text" in ds["train"].column_names:
            names = [n for _, n in sorted({(r[label_col], r["label_text"]) for r in ds["train"]})]
        remap = (lambda y: y) if names is None else {i: aliases[_norm_name(n)] for i, n in enumerate(names)}.__getitem__

        def conv(split):
            return [{"text": " ".join(str(r[text_col]).split()), "label": remap(r[label_col])} for r in ds[split]]

        return conv("train"), conv(test_split)
    raise RuntimeError("Could not load any mirror:\n  " + "\n  ".join(errors))


LOADERS = {
    # GLUE test labels are hidden -> sst2 is evaluated on its validation split (872)
    "sst2": lambda: _load_text_cls([("nyu-mll/glue", "sst2", "sentence", "label")],
                                   {"negative": 0, "positive": 1}),
    "cr": lambda: _load_text_cls([("SetFit/CR", None, "text", "label")], {"negative": 0, "positive": 1}),
    "subj": lambda: _load_text_cls([("SetFit/subj", None, "text", "label")], {"objective": 0, "subjective": 1}),
    "trec": lambda: _load_text_cls(
        [("CogComp/trec", None, "text", "coarse_label"), ("SetFit/TREC-QC", None, "text", "label_coarse")],
        {"abbr": 0, "abbreviation": 0, "enty": 1, "entity": 1, "desc": 2, "description": 2,
         "hum": 3, "human": 3, "loc": 4, "location": 4, "num": 5, "numeric": 5}),
    "emotion": lambda: _load_text_cls(
        [("dair-ai/emotion", "split", "text", "label"), ("SetFit/emotion", None, "text", "label")],
        {"sadness": 0, "joy": 1, "love": 2, "anger": 3, "fear": 4, "surprise": 5}),
    "ag_news": lambda: _load_text_cls([("fancyzhx/ag_news", None, "text", "label")],
                                      {"world": 0, "sports": 1, "business": 2, "sci/tech": 3}),
}


def make_synthetic(n_train: int = 2000, n_test: int = 500, seed: int = 0, variant: int = 0):
    """A rule that cannot be solved from pre-training priors: the label is set
    by which hidden word group the *first* nonsense word comes from."""
    rng = random.Random(seed + 1000 * variant)
    group_a = ["blick", "dax", "wug", "fep", "toma", "lorp"]
    group_b = ["zib", "mek", "gorp", "vun", "snerf", "plim"]
    if variant == 1:
        group_a, group_b = group_a[:3] + group_b[:3], group_a[3:] + group_b[3:]
    vocab = group_a + group_b

    def ex():
        label = rng.randint(0, 1)
        first = rng.choice(group_a if label == 0 else group_b)
        rest = [rng.choice(vocab) for _ in range(rng.randint(2, 4))]
        return {"text": " ".join([first] + rest), "label": label}

    return [ex() for _ in range(n_train)], [ex() for _ in range(n_test)]


LOADERS["synthetic"] = make_synthetic
LOADERS["synthetic2"] = lambda: make_synthetic(variant=1)
ALL_TASKS = sorted(LOADERS)


def load_task(name: str, n_test: int | None = 500, seed: int = 0, pool_size: int | None = None,
              dev_size: int = 0) -> ClassificationTask:
    """Load a task.

    * the test set is subsampled once with a fixed seed (same for every run);
    * ``dev_size`` > 0 carves a dev set out of the *training* split;
    * ``pool_size`` caps the remaining training pool demos are chosen from.
    """
    if name not in LOADERS:
        raise ValueError(f"Unknown task {name!r}. Available: {ALL_TASKS}")
    train, test = LOADERS[name]()
    rng = random.Random(seed + 1)
    train = list(train)
    rng.shuffle(train)
    dev, train = train[:dev_size], train[dev_size:]
    if pool_size is not None:
        train = train[:pool_size]
    if n_test is not None and n_test < len(test):
        test = random.Random(seed).sample(test, n_test)
    return ClassificationTask(name=name, templates=TEMPLATES[name], train=train, test=test, dev=dev)
