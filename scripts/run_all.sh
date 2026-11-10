#!/usr/bin/env bash
# The whole project in one script (single GPU, one model for everything).
#   1) analysis:   shots / labels / order on two tasks
#   2) method:     learn selection rules on meta-train tasks
#   3) evaluation: compare all selectors on unseen meta-test tasks (5 demo orders)
#   4) figures + tables
# Quick smoke test first (separate output folder, so it never mixes with real results):
#   OUT=smoke N_TEST=50 ROUNDS=2 DEV=50 bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-Qwen/Qwen3.5-2B}       # target model = selector
OPTIMIZER=${OPTIMIZER:-$MODEL}         # a larger model writes better rules, e.g. Qwen/Qwen3.5-9B
N_TEST=${N_TEST:-500}; ROUNDS=${ROUNDS:-10}; DEV=${DEV:-200}; BS=${BS:-16}
OUT=${OUT:-runs/main}                  # results/, rule_opt/ and figures/ all go here
ANALYSIS=${ANALYSIS:-"sst2 trec"}
META_TRAIN=${META_TRAIN:-"sst2 ag_news trec"}
META_TEST=${META_TEST:-"cr subj emotion"}
EMBEDDER=${EMBEDDER:-sentence-transformers/all-MiniLM-L6-v2}
RULES=$OUT/rule_opt

# 1) analysis
python scripts/run_experiment.py --model "$MODEL" --task $ANALYSIS --exp shots  --n-test "$N_TEST" --batch-size "$BS" --out "$OUT/results"
python scripts/run_experiment.py --model "$MODEL" --task $ANALYSIS --exp labels --n-test "$N_TEST" --batch-size "$BS" --out "$OUT/results"
python scripts/run_experiment.py --model "$MODEL" --task $ANALYSIS --exp order  --seeds 3 --n-test "$N_TEST" --batch-size "$BS" --out "$OUT/results"

# 2) learn rules (dev data comes from the training split of the meta-train tasks)
python scripts/optimize_rules.py --target "$MODEL" --optimizer "$OPTIMIZER" --tasks $META_TRAIN \
  --rounds "$ROUNDS" --dev-size "$DEV" --embedder "$EMBEDDER" --batch-size "$BS" --out "$RULES"

# 3) compare selectors on unseen tasks
python scripts/run_experiment.py --model "$MODEL" --task $META_TEST --exp method --rules "$RULES/best_rules.txt" \
  --orders 5 --embedder "$EMBEDDER" --n-test "$N_TEST" --batch-size "$BS" --out "$OUT/results"

# 4) $OUT/figures/*.png and $OUT/figures/tables.md
python scripts/plot_results.py --results "$OUT/results" --out "$OUT/figures" --rule-log "$RULES/log.jsonl"
