#!/usr/bin/env bash
set -euo pipefail

model="${1:-random}"
prompt_level="${2:-1}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

scenarios=(
  scene1_single_direct_or_random
  scene2_single_scrambled_fixed
  scene3_triad_delay_invert
)

for scenario in "${scenarios[@]}"; do
  "$ISAACSIM_ROOT/python.sh" "$script_dir/inference.py" \
    --scenario "$scenario" \
    --level "$prompt_level" \
    --model "$model" \
    --headless
done
