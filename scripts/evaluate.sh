#!/usr/bin/env bash
set -euo pipefail

model="${1:-random}"
level="${2:-1}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

scenarios=(
  triad_delay_invert
  quad_delay_swap_random
  quint_all_distractors
)

for scenario in "${scenarios[@]}"; do
  "$ISAACSIM_ROOT/python.sh" "$script_dir/inference.py" \
    --scenario "$scenario" \
    --level "$level" \
    --model "$model" \
    --headless
done
