#!/usr/bin/env bash
set -euo pipefail

model="${1:-random}"
prompt_level_arg="${2:-1}"
runs_per_scenario="${3:-4}"
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
  scene1_single_command_causality
  scene2_single_scrambled_stability
  scene3_triad_delay_invert
)

if [[ "$prompt_level_arg" == "all" ]]; then
  prompt_levels=(0 1 2 3)
elif [[ "$prompt_level_arg" =~ ^[0-3]$ ]]; then
  prompt_levels=("$prompt_level_arg")
else
  echo "Error: prompt level must be 0, 1, 2, 3, or all." >&2
  exit 1
fi

if [[ ! "$runs_per_scenario" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: runs_per_scenario must be a positive integer." >&2
  exit 1
fi

if ((runs_per_scenario % 4 != 0)); then
  echo "Error: runs_per_scenario must be divisible by 4 to balance scene 2 labels and option order." >&2
  exit 1
fi

if [[ ! "$start_seed" =~ ^[0-9]+$ ]]; then
  echo "Error: start_seed must be a non-negative integer." >&2
  exit 1
fi

num_scenarios="${#scenarios[@]}"

total_evaluations="$((
  runs_per_scenario * num_scenarios * ${#prompt_levels[@]}
))"
evaluation_number=0

for prompt_level in "${prompt_levels[@]}"; do
  for scenario in "${scenarios[@]}"; do
    for ((run_index = 0; run_index < runs_per_scenario; run_index++)); do
      seed="$((start_seed + run_index))"
      evaluation_number="$((evaluation_number + 1))"

      echo "[$evaluation_number/$total_evaluations] model=$model level=$prompt_level seed=$seed scenario=$scenario"
      "$python_bin" "$script_dir/inference.py" \
        --scenario "$scenario" \
        --level "$prompt_level" \
        --model "$model" \
        --seed "$seed" \
        --headless
    done
  done
done
