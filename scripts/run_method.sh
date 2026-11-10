#!/usr/bin/env bash
# Full method pipeline:
#   1) learn a selection rulebook on meta-train tasks
#   2) compare all selectors on unseen meta-test tasks (5 demo orders; order 0 = main table)
#   3) transfer: reuse the same rules + selector for other target-model sizes
#   4) plots and tables
# Model ids are variables: check the exact names on the Hub before running.
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET=${TARGET:-Qwen/Qwen3.5-2B}
SELECTOR=${SELECTOR:-$TARGET}
OPTIMIZER=${OPTIMIZER:-$SELECTOR}          # e.g. a larger Qwen3.5, or use OPTIMIZER_BACKEND=api
OPTIMIZER_BACKEND=${OPTIMIZER_BACKEND:-hf}
TRANSFER_TARGETS=${TRANSFER_TARGETS:-"Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-4B"}
META_TRAIN=${META_TRAIN:-"sst2 ag_news trec"}
META_TEST=${META_TEST:-"cr subj emotion rte"}  # add banking77 with a smaller N_TEST (77 labels are slow to score)
K=${K:-8}; TOP_M=${TOP_M:-20}; N_TEST=${N_TEST:-500}; ORDERS=${ORDERS:-5}; BS=${BS:-16}
RULES_DIR=${RULES_DIR:-rule_opt/$(basename "$TARGET")}
OUT=${OUT:-results}

python scripts/optimize_rules.py --target "$TARGET" --selector "$SELECTOR" --optimizer "$OPTIMIZER" \
  --optimizer-backend "$OPTIMIZER_BACKEND" --tasks $META_TRAIN --rounds 10 --dev-size 200 \
  --k "$K" --top-m "$TOP_M" --batch-size "$BS" --out "$RULES_DIR"

python scripts/run_experiment.py --model "$TARGET" --selector-model "$SELECTOR" --task $META_TEST --exp method \
  --rules "$RULES_DIR/best_rules.txt" --k "$K" --top-m "$TOP_M" --orders "$ORDERS" --n-test "$N_TEST" \
  --batch-size "$BS" --out "$OUT"

for M in $TRANSFER_TARGETS; do
  python scripts/run_experiment.py --model "$M" --selector-model "$SELECTOR" --task $META_TEST --exp method \
    --rules "$RULES_DIR/best_rules.txt" --k "$K" --top-m "$TOP_M" --orders 1 --n-test "$N_TEST" \
    --batch-size "$BS" --out "$OUT"
done

python scripts/plot_results.py --results "$OUT" --out figures --rule-log "$RULES_DIR/log.jsonl"
