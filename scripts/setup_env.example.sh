#!/usr/bin/env bash

# Source this file to set up the environment variables for running the benchmark.
# source scripts/setup_env.sh

export OPENAI_API_KEY="${OPENAI_API_KEY:-your_api_key_here}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1/}"
export ISAACSIM_ROOT="${ISAACSIM_ROOT:-your_isaacsim_root_here}"

export API_KEY="$OPENAI_API_KEY"
export BASE_URL="$OPENAI_BASE_URL"

if [[ ! -d "$ISAACSIM_ROOT" ]]; then
    echo "Warning: ISAACSIM_ROOT does not exist. Please set it to your Isaac Sim installation path." >&2
fi

echo "ISAACSIM_ROOT=$ISAACSIM_ROOT"
echo "OPENAI_BASE_URL=$OPENAI_BASE_URL"
if [[ -z "$OPENAI_API_KEY" || "$OPENAI_API_KEY" == "your_api_key_here" ]]; then
    echo "Warning: OPENAI_API_KEY is not set. Please set it before evaluating API models."
fi
