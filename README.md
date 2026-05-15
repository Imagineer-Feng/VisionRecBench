# VisionRecBench: Visual Self-Recognition with Mimic Robot Arms

VisionRecBench is a standalone Isaac Sim benchmark for embodied self-recognition. Each episode places several visually similar robotic arms in one scene. One candidate is the target agent's own arm; the other candidates are distractors that imitate the target through delayed, inverted, shuffled, smoothed, or random motor commands.

The agent receives:

- the current RGB observation,
- a short visual history,
- the motor-command history sent to its own arm,
- and the set of candidate arm IDs ordered from left to right.

The task is to answer which visible candidate arm is itself.

## Project Layout

```text
VisionRecBench/
  source/
    action.py               # candidate answer schema
    agent.py                # OpenAI-compatible and random candidate identifiers
    env.py                  # Isaac Sim procedural multi-arm environment
    preprocess.py           # scenario/config loading
    prompts.py              # level 0-3 task prompts
  tasks/
    arm_repo.json           # procedural arm geometry and colors
    distractor_repo.json    # mimic behavior definitions
    scenario_repo.json      # benchmark scenarios
  scripts/
    inference.py            # single-run evaluation entrypoint
    evaluate.sh             # batch evaluation helper
    setup_env.sh            # local environment variables
```

## Setup

Install Isaac Sim and set `ISAACSIM_ROOT`. For API models, also set `OPENAI_API_KEY`.

```shell
cd VisionRecBench
source scripts/setup_env.sh
$ISAACSIM_ROOT/python.sh -m pip install openai==1.79.0
```

`agent.py` also accepts the legacy variables `API_KEY` and `BASE_URL`.

## Single Scenario

```shell
cd VisionRecBench
$ISAACSIM_ROOT/python.sh scripts/inference.py \
  --scenario triad_delay_invert \
  --level 1 \
  --model gpt-4o \
  --headless
```

Random baseline:

```shell
$ISAACSIM_ROOT/python.sh scripts/inference.py \
  --scenario triad_delay_invert \
  --level 1 \
  --model random \
  --headless
```

Franka Panda arm scenario:

```shell
$ISAACSIM_ROOT/python.sh scripts/inference.py \
  --scenario triad_panda_delay_invert \
  --level 1 \
  --model random \
  --headless
```

The default run uses `PathTracing`, 1024x1024 observations, 16 samples per pixel, and enough render frames for the camera buffer to settle. This is intentionally slower than the old fast defaults because it avoids the noisy, blurry `RayTracedLighting` output.

Isaac Sim may spend one or two minutes after `app ready` compiling shaders and initializing render buffers, especially on the first run in a fresh environment. The many `omni.isaac.* has been deprecated` warnings are emitted by Isaac Sim extensions and are not VisionRecBench errors. A healthy run eventually prints per-step progress and writes outputs to `logs/` and `results/`.

## Full Evaluation

```shell
cd VisionRecBench
chmod +x scripts/evaluate.sh
./scripts/evaluate.sh gpt-4o 1
```

Each run writes observations and logs to `logs/<timestamp>/` and metrics to `results/level*/<model>/<scenario>/`.

## Metrics

The result JSON reports:

- `accuracy`: fraction of episode steps where the model selected the target arm,
- `final_correct`: whether the final answer was correct,
- `majority_correct`: whether the majority selected candidate was the target,
- `first_correct_step`: first step where the target was identified,
- `bad_response`: unparsable or out-of-range answers.

The core score is `accuracy`; `final_correct` and `majority_correct` are useful when treating an episode as a single identification problem.
