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


def make_synthetic(n_train: int = 2000, n_test: int = 500, seed: int = 0):
    """A task that cannot be solved from pre-training priors.

    Inputs are sequences of nonsense words; the label is determined by which
    of two hidden word groups the *first* word comes from. The rule can only
    be inferred from demonstrations. Also used for offline tests.
    """
    rng = random.Random(seed)
    group_a = ["blick", "dax", "wug", "fep", "toma", "lorp"]
    group_b = ["zib", "mek", "gorp", "vun", "snerf", "plim"]
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
}

CLS_LOADERS = {
    "sst2": load_sst2,
    "ag_news": load_ag_news,
    "rte": load_rte,
    "synthetic": make_synthetic,
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


def load_task(name: str, n_test: int | None = 500, seed: int = 0):
    """Load a task; the test set is subsampled once with a fixed seed so that
    every condition / model is evaluated on exactly the same examples."""
    if name in CLS_LOADERS:
        train, test = CLS_LOADERS[name]()
        return ClassificationTask(
            name=name,
            templates=TEMPLATES[name],
            train=train,
            test=_subsample(test, n_test, seed),
            text_fields=TEXT_FIELDS[name],
        )
    if name in GEN_LOADERS:
        train, test = GEN_LOADERS[name]()
        return GenerationTask(name=name, train=train, test=_subsample(test, n_test, seed))
    raise ValueError(f"Unknown task {name!r}. Available: {ALL_TASKS}")
