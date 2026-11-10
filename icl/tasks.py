"""Task definitions and dataset loading.

Every task exposes the same small interface:

* ``train``: pool that demonstrations are sampled from
* ``test``:  evaluation examples
* classification tasks: ``templates`` (prompt format + label words)
* generation tasks (GSM8K): ``format_question`` / ``format_answer``

Examples are plain dicts. Classification examples always carry an integer
``label``; the remaining keys are the text fields referenced by templates.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

# Label words used for the "abstract labels" condition (Min et al. 2022 style):
# semantically empty tokens, so the model can only use them if it learns the
# mapping from the demonstrations.
ABSTRACT_LABELS = ["foo", "bar", "baz", "qux", "zap", "wug"]


@dataclass
class Template:
    """How one example is rendered.

    ``input_fmt`` is a ``str.format`` pattern over the example's fields and
    ends right before the label. The label is appended as ``" " + word``.
    """

    input_fmt: str
    verbalizers: list[str]
    sep: str = "\n\n"
    instruction: str = ""  # optional text placed before all demos

    def render_input(self, ex: dict) -> str:
        return self.input_fmt.format(**ex)

    @staticmethod
    def render_label(word: str) -> str:
        return " " + word


@dataclass
class ClassificationTask:
    name: str
    templates: list[Template]
    train: list[dict]
    test: list[dict]
    text_fields: list[str]
    type: str = "classification"
    dev: list[dict] = field(default_factory=list)

    @property
    def num_labels(self) -> int:
        return len(self.templates[0].verbalizers)

    def content_free(self) -> dict:
        """Content-free input for contextual calibration (Zhao et al. 2021)."""
        return {f: "N/A" for f in self.text_fields}


@dataclass
class GenerationTask:
    """Math word problems with a numeric final answer (GSM8K-style)."""

    name: str
    train: list[dict]  # keys: question, rationale, answer
    test: list[dict]
    type: str = "generation"
    dev: list[dict] = field(default_factory=list)
    sep: str = "\n\n"
    stop: list[str] = field(default_factory=lambda: ["\n\nQuestion:", "\nQuestion:"])

    @staticmethod
    def format_question(q: str) -> str:
        return f"Question: {q}\nAnswer:"

    @staticmethod
    def format_answer(rationale: str | None, answer: str) -> str:
        if rationale:
            return f" {rationale} The answer is {answer}."
        return f" The answer is {answer}."


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------


def _hf(path: str, name: str | None = None):
    from datasets import load_dataset

    return load_dataset(path, name)


def _subsample(rows: list[dict], n: int | None, seed: int) -> list[dict]:
    if n is None or n >= len(rows):
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, n)


def load_sst2() -> tuple[list[dict], list[dict]]:
    ds = _hf("nyu-mll/glue", "sst2")  # GLUE test labels are hidden -> use validation
    conv = lambda d: [{"sentence": r["sentence"].strip(), "label": r["label"]} for r in d]
    return conv(ds["train"]), conv(ds["validation"])


def load_ag_news() -> tuple[list[dict], list[dict]]:
    ds = _hf("fancyzhx/ag_news")
    conv = lambda d: [{"text": " ".join(r["text"].split()), "label": r["label"]} for r in d]
    return conv(ds["train"]), conv(ds["test"])


def load_rte() -> tuple[list[dict], list[dict]]:
    ds = _hf("nyu-mll/glue", "rte")
    conv = lambda d: [
        {"sentence1": r["sentence1"].strip(), "sentence2": r["sentence2"].strip(), "label": r["label"]}
        for r in d
    ]
    return conv(ds["train"]), conv(ds["validation"])


# ---- generic loader for single-text classification datasets ---------------


def _label_names(ds_split, label_col: str):
    feat = ds_split.features.get(label_col)
    if feat is not None and hasattr(feat, "names"):
        return list(feat.names)
    return None


def _load_text_cls(candidates, aliases: dict[str, int] | None):
    """Try several HF dataset ids in order (mirrors move around).

    Each candidate is (path, config, text_col, label_col). Labels are mapped to
    our canonical order through their *names* (``aliases``: lowercased name ->
    index) so that a mirror with a different label order cannot silently
    scramble the verbalizers. With ``aliases=None`` the dataset's own label
    order/names are used and returned.
    """
    errors = []
    for path, cfg, text_col, label_col in candidates:
        try:
            ds = _hf(path, cfg)
        except Exception as e:  # noqa: BLE001 - try the next mirror
            errors.append(f"{path}: {type(e).__name__}: {e}")
            continue
        test_split = "test" if "test" in ds else "validation"
        names = _label_names(ds["train"], label_col)
        if names is None and "label_text" in ds["train"].column_names:
            pairs = sorted({(r[label_col], r["label_text"]) for r in ds["train"]})
            names = [n for _, n in pairs]
        if aliases is not None:
            if names is None:
                remap = lambda y: y  # assume canonical order
            else:
                table = {i: aliases[_norm_name(n)] for i, n in enumerate(names)}
                remap = table.__getitem__
            out_names = None
        else:
            remap, out_names = (lambda y: y), names

        def conv(split):
            return [{"text": " ".join(str(r[text_col]).split()), "label": remap(r[label_col])} for r in ds[split]]

        return conv("train"), conv(test_split), out_names
    raise RuntimeError("Could not load any mirror:\n  " + "\n  ".join(errors))


def _norm_name(n: str) -> str:
    return str(n).strip().lower().replace("_", " ")


def load_trec():
    aliases = {"abbr": 0, "abbreviation": 0, "enty": 1, "entity": 1, "desc": 2, "description": 2,
               "hum": 3, "human": 3, "loc": 4, "location": 4, "num": 5, "numeric": 5}
    tr, te, _ = _load_text_cls([("CogComp/trec", None, "text", "coarse_label"),
                                ("SetFit/TREC-QC", None, "text", "label_coarse")], aliases)
    return tr, te


def load_cr():
    tr, te, _ = _load_text_cls([("SetFit/CR", None, "text", "label")], {"negative": 0, "positive": 1})
    return tr, te


def load_subj():
    tr, te, _ = _load_text_cls([("SetFit/subj", None, "text", "label")], {"objective": 0, "subjective": 1})
    return tr, te


def load_emotion():
    aliases = {"sadness": 0, "joy": 1, "love": 2, "anger": 3, "fear": 4, "surprise": 5}
    tr, te, _ = _load_text_cls([("dair-ai/emotion", "split", "text", "label"),
                                ("SetFit/emotion", None, "text", "label")], aliases)
    return tr, te


def load_banking77():
    tr, te, names = _load_text_cls([("mteb/banking77", None, "text", "label"),
                                    ("PolyAI/banking77", None, "text", "label")], None)
    if names is None:
        raise RuntimeError("banking77 mirror did not provide label names")
    words = [_norm_name(n) for n in names]
    TEMPLATES["banking77"] = [
        Template("Query: {text}\nIntent:", words),
        Template("{text}\nThe customer is asking about", words),
        Template("Input: {text}\nOutput:", words),
    ]
    return tr, te


def make_synthetic(n_train: int = 2000, n_test: int = 500, seed: int = 0, variant: int = 0):
    """A task that cannot be solved from pre-training priors.

    Inputs are sequences of nonsense words; the label is determined by which
    of two hidden word groups the *first* word comes from. The rule can only
    be inferred from demonstrations. Also used for offline tests.
    """
    rng = random.Random(seed + 1000 * variant)
    group_a = ["blick", "dax", "wug", "fep", "toma", "lorp"]
    group_b = ["zib", "mek", "gorp", "vun", "snerf", "plim"]
    if variant == 1:  # a second, unrelated synthetic task (used for meta-train/test splits in tests)
        group_a, group_b = group_a[:3] + group_b[:3], group_a[3:] + group_b[3:]
    vocab = group_a + group_b

    def ex():
        label = rng.randint(0, 1)
        first = rng.choice(group_a if label == 0 else group_b)
        rest = [rng.choice(vocab) for _ in range(rng.randint(2, 4))]
        return {"text": " ".join([first] + rest), "label": label}

    return [ex() for _ in range(n_train)], [ex() for _ in range(n_test)]


TEMPLATES: dict[str, list[Template]] = {
    "sst2": [
        Template("Review: {sentence}\nSentiment:", ["negative", "positive"]),
        Template("{sentence} It was", ["terrible", "great"]),
        Template("Input: {sentence}\nOutput:", ["negative", "positive"]),
    ],
    "ag_news": [
        Template("Article: {text}\nTopic:", ["World", "Sports", "Business", "Technology"]),
        Template("{text}\nThis news is about", ["world", "sports", "business", "science"]),
        Template("Input: {text}\nOutput:", ["World", "Sports", "Business", "Technology"]),
    ],
    "rte": [
        # GLUE RTE: 0 = entailment, 1 = not_entailment
        Template("{sentence1}\nQuestion: {sentence2} True or False?\nAnswer:", ["True", "False"]),
        Template("Premise: {sentence1}\nHypothesis: {sentence2}\nEntailment:", ["yes", "no"]),
        Template("Input: {sentence1} [SEP] {sentence2}\nOutput:", ["True", "False"]),
    ],
    "trec": [
        Template("Question: {text}\nAnswer type:", ["abbreviation", "entity", "description", "person", "location", "number"]),
        Template("{text}\nThis question asks for a", ["abbreviation", "thing", "description", "person", "place", "number"]),
        Template("Input: {text}\nOutput:", ["abbreviation", "entity", "description", "person", "location", "number"]),
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
    "emotion": [
        Template("Text: {text}\nEmotion:", ["sadness", "joy", "love", "anger", "fear", "surprise"]),
        Template("{text}\nThe writer feels", ["sad", "happy", "loving", "angry", "afraid", "surprised"]),
        Template("Input: {text}\nOutput:", ["sadness", "joy", "love", "anger", "fear", "surprise"]),
    ],
    "synthetic2": [
        Template("Input: {text}\nOutput:", ["A", "B"]),
        Template("{text} =>", ["left", "right"]),
        Template("Sequence: {text}\nClass:", ["red", "blue"]),
    ],
    "synthetic": [
        Template("Input: {text}\nOutput:", ["A", "B"]),
        Template("{text} =>", ["left", "right"]),
        Template("Sequence: {text}\nClass:", ["red", "blue"]),
    ],
}

TEXT_FIELDS = {
    "sst2": ["sentence"],
    "ag_news": ["text"],
    "rte": ["sentence1", "sentence2"],
    "synthetic": ["text"],
    "synthetic2": ["text"],
    "trec": ["text"],
    "cr": ["text"],
    "subj": ["text"],
    "emotion": ["text"],
    "banking77": ["text"],
}

CLS_LOADERS = {
    "sst2": load_sst2,
    "ag_news": load_ag_news,
    "rte": load_rte,
    "synthetic": make_synthetic,
    "synthetic2": lambda: make_synthetic(variant=1),
    "trec": load_trec,
    "cr": load_cr,
    "subj": load_subj,
    "emotion": load_emotion,
    "banking77": load_banking77,
}

_CALC = re.compile(r"<<.*?>>")


def _parse_gsm8k(r: dict) -> dict:
    rationale, answer = r["answer"].split("####")
    rationale = _CALC.sub("", rationale).strip().replace("\n", " ")
    return {"question": r["question"].strip(), "rationale": rationale, "answer": answer.strip().replace(",", "")}


def load_gsm8k():
    ds = _hf("openai/gsm8k", "main")
    return [_parse_gsm8k(r) for r in ds["train"]], [_parse_gsm8k(r) for r in ds["test"]]


def make_synthetic_math(n_train: int = 200, n_test: int = 50, seed: int = 0):
    """Tiny arithmetic word problems; offline stand-in for GSM8K in tests."""
    rng = random.Random(seed)

    def ex():
        a, b, c = rng.randint(2, 20), rng.randint(2, 20), rng.randint(1, 5)
        q = f"Tom has {a} apples and buys {b} more. He gives away {c}. How many apples does he have?"
        r = f"Tom starts with {a} apples. After buying {b} he has {a + b}. After giving away {c} he has {a + b - c}."
        return {"question": q, "rationale": r, "answer": str(a + b - c)}

    return [ex() for _ in range(n_train)], [ex() for _ in range(n_test)]


GEN_LOADERS = {"gsm8k": load_gsm8k, "synthetic_math": make_synthetic_math}

ALL_TASKS = sorted(list(CLS_LOADERS) + list(GEN_LOADERS))


def load_task(name: str, n_test: int | None = 500, seed: int = 0, pool_size: int | None = None,
              dev_size: int = 0):
    """Load a task.

    * the test set is subsampled once with a fixed seed, so every condition /
      model is evaluated on exactly the same examples;
    * ``dev_size`` > 0 carves a held-out dev set out of the *training* split
      (used by rule optimisation; never touches the test split);
    * ``pool_size`` caps the remaining training pool that demonstrations are
      selected from (keeps retrieval and selection cheap).
    """
    if name in CLS_LOADERS:
        train, test = CLS_LOADERS[name]()
    elif name in GEN_LOADERS:
        train, test = GEN_LOADERS[name]()
    else:
        raise ValueError(f"Unknown task {name!r}. Available: {ALL_TASKS}")
    rng = random.Random(seed + 1)
    train = list(train)
    rng.shuffle(train)
    dev, train = train[:dev_size], train[dev_size:]
    if pool_size is not None:
        train = train[:pool_size]
    test = _subsample(test, n_test, seed)
    if name in CLS_LOADERS:
        task = ClassificationTask(name=name, templates=TEMPLATES[name], train=train, test=test,
                                  text_fields=TEXT_FIELDS[name])
    else:
        task = GenerationTask(name=name, train=train, test=test)
    task.dev = dev
    return task
