#!/usr/bin/env bash
set -euo pipefail

model="${1:-random}"
prompt_level_arg="${2:-1}"
runs_per_scenario="${3:-12}"
start_seed="${4:-1}"
resume=0
only_scenario=""

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/.." && pwd)"

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
  scene3_dyad_causal_identification
)

extra_args=("${@:5}")
arg_index=0
while ((arg_index < ${#extra_args[@]})); do
  arg="${extra_args[$arg_index]}"
  case "$arg" in
    --resume)
      resume=1
      ;;
    --scenario)
      arg_index="$((arg_index + 1))"
      if ((arg_index >= ${#extra_args[@]})); then
        echo "Error: --scenario requires a scenario name." >&2
        exit 1
      fi
      only_scenario="${extra_args[$arg_index]}"
      ;;
    --scenario=*)
      only_scenario="${arg#--scenario=}"
      ;;
    *)
      echo "Error: unknown argument: $arg" >&2
      exit 1
      ;;
  esac
  arg_index="$((arg_index + 1))"
done

selected_scenarios=("${scenarios[@]}")
if [[ -n "$only_scenario" ]]; then
  selected_scenarios=()
  for scenario in "${scenarios[@]}"; do
    if [[ "$scenario" == "$only_scenario" ]]; then
      selected_scenarios=("$scenario")
      break
    fi
  done
  if ((${#selected_scenarios[@]} == 0)); then
    echo "Error: unknown scenario: $only_scenario" >&2
    exit 1
  fi
fi

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

if ((runs_per_scenario % 12 != 0)); then
  echo "Error: runs_per_scenario must be divisible by 12 to balance scene 1/2 binary labels and scene 3 target/role positions." >&2
  exit 1
fi

if [[ ! "$start_seed" =~ ^[0-9]+$ ]]; then
  echo "Error: start_seed must be a non-negative integer." >&2
  exit 1
fi

completed_index=""
if ((resume)); then
  completed_index="$(mktemp)"
  trap 'rm -f "$completed_index"' EXIT

  if command -v python3 >/dev/null 2>&1; then
    index_python="python3"
  else
    index_python="$python_bin"
  fi

  "$index_python" - "$repo_dir" >"$completed_index" <<'PY'
import json
import pathlib
import sys

repo_dir = pathlib.Path(sys.argv[1])
for args_file in (repo_dir / "logs").glob("*/args.json"):
    try:
        args = json.loads(args_file.read_text())
    except Exception:
        continue

    scenario = args.get("scenario")
    prompt_level = args.get("prompt_level")
    model = args.get("model")
    seed = args.get("seed")
    if scenario is None or prompt_level is None or model is None or seed is None:
        continue

    tag = args_file.parent.name
    model_dir = str(model).replace("/", "-")
    result_pattern = (
        f"scene*/prompt_level{prompt_level}/{model_dir}/{scenario}/{tag}.json"
    )
    if any((repo_dir / "results").glob(result_pattern)):
        print(f"{prompt_level}\t{model}\t{scenario}\t{seed}")
PY
fi

completed_run_exists() {
  local prompt_level="$1"
  local scenario="$2"
  local seed="$3"
  local key

  if [[ -z "$completed_index" ]]; then
    return 1
  fi

  key="$(printf '%s\t%s\t%s\t%s' "$prompt_level" "$model" "$scenario" "$seed")"
  grep -Fxq "$key" "$completed_index"
}

num_scenarios="${#selected_scenarios[@]}"

total_evaluations="$((
  runs_per_scenario * num_scenarios * ${#prompt_levels[@]}
))"
evaluation_number=0

for prompt_level in "${prompt_levels[@]}"; do
  for scenario in "${selected_scenarios[@]}"; do
    for ((run_index = 0; run_index < runs_per_scenario; run_index++)); do
      seed="$((start_seed + run_index))"
      evaluation_number="$((evaluation_number + 1))"

      if ((resume)) && completed_run_exists "$prompt_level" "$scenario" "$seed"; then
        echo "[$evaluation_number/$total_evaluations] skipping completed model=$model level=$prompt_level seed=$seed scenario=$scenario"
        continue
      fi

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
