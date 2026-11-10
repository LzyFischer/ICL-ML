# ICL-ML: Demonstration Selection for In-Context Learning

A small class-project framework with two parts:

1. **Analysis**: how much do the number, labels and order of demonstrations affect in-context learning (ICL)?
2. **Method**: an LLM picks demonstrations for each query by following a set of natural-language *rules*, and a feedback loop automatically improves those rules. The learned rules are then applied, frozen, to unseen tasks.

Everything is inference only. By default a single model (Qwen3.5-2B) plays every role, so one GPU is enough.

## Quick start

```bash
pip install -r requirements.txt

# Smoke test first: small scale, separate output folder, finishes in minutes
OUT=runs/smoke N_TEST=50 ROUNDS=2 DEV=50 bash scripts/run_all.sh

# Full run
bash scripts/run_all.sh
```

All outputs go under `runs/main/`:

```
runs/main/results/<exp>/<task>/<model>/summary.csv   one row per experimental condition
runs/main/rule_opt/best_rules.txt                     the learned rules
runs/main/rule_opt/log.jsonl                          rules, score and raw optimizer reply for every round
runs/main/figures/*.png, tables.md                    figures and tables
```

Settings you can change through environment variables:

- `MODEL`: target model, also used as the selector
- `OPTIMIZER`: the model that writes the rules. A larger model usually writes better rules, e.g. `OPTIMIZER=Qwen/Qwen3.5-9B`
- `N_TEST`, `ROUNDS`, `DEV`, `BS` (batch size), `OUT`

## Datasets

Only short-text classification tasks with few labels, so everything runs fast:

| Task | Labels | Used for |
|---|---|---|
| `sst2` (movie review sentiment) | 2 | analysis + meta-train |
| `ag_news` (news topic) | 4 | meta-train |
| `trec` (question type) | 6 | analysis + meta-train |
| `cr` (product review sentiment) | 2 | meta-test |
| `subj` (subjective / objective) | 2 | meta-test |
| `emotion` (emotion) | 6 | meta-test |

Datasets are downloaded from HuggingFace automatically. Each task tries several mirrors in turn and maps labels to a fixed order **by label name**, so a mirror with a different label order cannot silently scramble the label words.

The dataset IDs were not test-downloaded when this code was written. If one fails to load, the error lists the reason for every mirror; fix the path in `icl/tasks.py`.

`synthetic` / `synthetic2` are locally generated tasks used only by the offline tests.

## Part 1: Analysis experiments

| `--exp` | Question |
|---|---|
| `shots` | How does accuracy change with the number of demonstrations k (0–32)? |
| `labels` | What happens when demo labels are random, flipped, or abstract (foo, bar)? For flipped labels we also report how often the model follows the flipped mapping (`acc_flip`). |
| `order` | How much does accuracy vary across 10 orderings of the same demos, compared with switching to a different demo set? |
| `template` | What happens when the prompt template changes? (not run by `run_all.sh`) |

Predictions come from comparing the log-probability of each label word, so no output parsing is needed. Every condition is evaluated on the same fixed test examples. Each run also reports two calibrated accuracies: `acc_cal` (contextual calibration) and `acc_bc` (batch calibration).

## Part 2: Method

**Per-query selection in two steps:**

1. A sentence-embedding model retrieves the 20 training examples most similar to the query as candidates.
2. The selector LLM judges each candidate separately: "According to these rules, is this candidate a good demonstration for the query?" The probability of answering Yes is the candidate's score. It is combined with the retrieval similarity, and the 8 top-scoring candidates are used.

**Learning the rules** (`scripts/optimize_rules.py`). On the meta-train tasks, each round:

1. selects demos for the dev queries with the current rules. The dev set is carved out of the training split; the test set is never used;
2. evaluates the target model on the dev set. The score is the accuracy gain over kNN;
3. gives the optimizer the past rules with their scores plus diagnostics (wins and losses against kNN, statistics of the chosen demos, the model's prediction distribution), and the optimizer rewrites the rules;
4. keeps the new rules if they score higher.

At the end, the learned rules are **frozen** and evaluated on the unseen meta-test tasks.

**Methods compared:**

| Method | Purpose |
|---|---|
| `random` | random selection |
| `knn` | the k most similar examples; **the key baseline** |
| `topm_random` | k random examples from the top 20; rules out "the candidate pool is simply better" |
| `llm_norules` | selector without rules |
| `llm_human` | hand-written rules (also the starting point of rule optimization) |
| `llm_learned` | learned rules; **our method** |
| `llm_learned_adaptive` | uses kNN directly when the retrieved neighbours all share one label, skipping the LLM (`llm_frac` records how often the LLM is called) |

The main table uses demo order seed 0 only (`method_acc.png`, `tables.md`). `method_orders_acc.png` shows every method under 5 demo orders, to show that the conclusions do not depend on the order. `rule_opt_curve.png` shows the rule-optimization progress.

## Running individual steps

```bash
python scripts/run_experiment.py --model Qwen/Qwen3.5-2B --task sst2 trec --exp shots
python scripts/optimize_rules.py --target Qwen/Qwen3.5-2B --tasks sst2 ag_news trec --out runs/main/rule_opt
python scripts/run_experiment.py --model Qwen/Qwen3.5-2B --task cr subj emotion --exp method \
    --rules runs/main/rule_opt/best_rules.txt --orders 5
python scripts/plot_results.py --results results --out figures
```

The optimizer can also be an API model: `--optimizer-backend api --optimizer <model name> --base-url ...` (requires `OPENAI_API_KEY`).

## Code layout

```
icl/tasks.py       datasets and templates
icl/prompts.py     demo sampling, label manipulation, prompt construction
icl/models.py      HFModel (label scoring + chat generation), APIModel (optimizer only)
icl/runner.py      evaluation loops for the analysis and method experiments; writes summary.csv (resumable)
icl/retrieval.py   sentence-embedding retrieval
icl/selection.py   selectors
icl/rule_opt.py    rule optimization loop
scripts/           run_all.sh / run_experiment.py / optimize_rules.py / plot_results.py
tests/             offline tests with a tiny randomly initialized GPT-2, no network needed: python -m pytest -q
```

## Notes

- Check the model ID on HuggingFace first. If Qwen3.5 is published as an image-text (multimodal) checkpoint, the code automatically loads it with the matching class and feeds text only. Qwen3.5 needs a recent version of transformers.
- If `rule_opt_curve.png` shows that no new rules were produced in any round, open `log.jsonl` and read the optimizer's raw replies. Usually the model did not follow the `<rules>...</rules>` format; switching to a larger `OPTIMIZER` fixes this.
- The selector looks at one candidate at a time, so it cannot guarantee diversity within a set of demos. This is worth mentioning as a limitation in the report.
