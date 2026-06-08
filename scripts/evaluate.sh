#!/usr/bin/env bash
set -euo pipefail

model="${1:-random}"
prompt_level="${2:-1}"
total_runs="${3:-3}"
start_seed="${4:-1}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="$PYTHON_BIN"
elif [[ -n "${ISAACSIM_ROOT:-}" ]]; then
  python_bin="$ISAACSIM_ROOT/python.sh"
else
  echo "Error: set PYTHON_BIN or ISAACSIM_ROOT before running evaluation." >&2
  exit 1
fi

if [[ ! -x "$python_bin" ]]; then
  echo "Error: Python executable not found or not executable: $python_bin" >&2
  exit 1
fi

scenarios=(
  scene1_single_direct_or_random
  scene2_single_scrambled_fixed
  scene3_triad_delay_invert
)

num_scenarios="${#scenarios[@]}"

for ((run_index = 0; run_index < total_runs; run_index++)); do
  scenario="${scenarios[$((run_index % num_scenarios))]}"
  seed="$((start_seed + run_index / num_scenarios))"
  run_number="$((run_index + 1))"

  echo "[$run_number/$total_runs] model=$model level=$prompt_level seed=$seed scenario=$scenario"
  "$python_bin" "$script_dir/inference.py" \
    --scenario "$scenario" \
    --level "$prompt_level" \
    --model "$model" \
    --seed "$seed" \
    --headless
done
