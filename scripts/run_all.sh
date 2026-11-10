#!/usr/bin/env bash
# Suggested full run for the class project: 1 classification + 1 reasoning task,
# 3 model sizes. Edit MODELS / TASKS to fit your GPU budget.
#
# Rough cost on one 24GB GPU with n-test=500:
#   0.5B / 1.5B: minutes per experiment; 7B: ~an hour for `shots` up to k=32.
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=${MODELS:-"Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-7B"}
CLS_TASKS=${CLS_TASKS:-"sst2 ag_news"}
OUT=${OUT:-results}
BS=${BS:-8}

for M in $MODELS; do
  # --- classification -------------------------------------------------
  python scripts/run_experiment.py --model "$M" --task $CLS_TASKS --exp shots    --shots 0 1 2 4 8 16 32 --seeds 5 --batch-size "$BS" --out "$OUT"
  python scripts/run_experiment.py --model "$M" --task $CLS_TASKS --exp labels   --k 16 --seeds 5          --batch-size "$BS" --out "$OUT"
  python scripts/run_experiment.py --model "$M" --task $CLS_TASKS --exp order    --k 8  --seeds 3 --perms 10 --batch-size "$BS" --out "$OUT"
  python scripts/run_experiment.py --model "$M" --task $CLS_TASKS --exp template --k 16 --seeds 3          --batch-size "$BS" --out "$OUT"
  # --- reasoning (generation is slower: fewer examples / seeds) ---------
  python scripts/run_experiment.py --model "$M" --task gsm8k --exp shots --shots 0 1 2 4 8 --seeds 2 --n-test 300 --batch-size "$BS" --out "$OUT"
  python scripts/run_experiment.py --model "$M" --task gsm8k --exp cot   --k 4 --seeds 2           --n-test 300 --batch-size "$BS" --out "$OUT"
done

python scripts/plot_results.py --results "$OUT" --out figures
python scripts/plot_results.py --results "$OUT" --out figures/calibrated --metric acc_cal
