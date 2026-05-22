# VisionRecBench: Visual Self-Recognition with Mimic Robot Arms

VisionRecBench is a standalone Isaac Sim benchmark for embodied self-recognition. It now supports three experimental levels:

- Experiment level 1: one visible arm, binary self/non-self judgment. The arm either follows the motor command directly or moves with random independent commands.
- Experiment level 2: one visible arm under a fixed scrambled action-space mapping. A repeated fixed command can produce a stable but non-standard action, which should still be recognized as self-motion.
- Experiment level 3: the original multi-arm visual self-recognition task. One candidate arm is the target agent's own arm; the other candidates imitate the target through delayed, inverted, shuffled, smoothed, or random motor commands.

The agent receives:

- the current RGB observation,
- a short visual history,
- the motor-command history available to the agent,
- and either binary self/non-self answer options or candidate arm IDs ordered from left to right.

The task is to answer whether the visible arm is itself, or which visible candidate arm is itself, depending on the scenario.

## Project Layout

```text
VisionRecBench/
  source/
    action.py               # candidate answer schema
    agent.py                # OpenAI-compatible and random candidate identifiers
    env.py                  # Isaac Sim multi-arm environment and Panda/USD loading
    preprocess.py           # scenario/config loading
    prompts.py              # prompt difficulty levels 0-3
    render_config.py        # fixed renderer, resolution, and sampling settings
  tasks/
    arm_repo.json           # procedural and Panda arm definitions
    distractor_repo.json    # mimic behavior definitions
    scenario_repo.json      # benchmark scenarios
  scripts/
    inference.py            # single-run evaluation entrypoint
    evaluate.sh             # batch evaluation helper
    setup_env.example.sh    # example local environment variables
    setup_env.sh            # local environment variables, if created
  logs/                     # per-run observations and logs, generated at runtime
  results/                  # per-run metrics, generated at runtime
```

## Setup

Install Isaac Sim and set `ISAACSIM_ROOT`. For API models, also set `OPENAI_API_KEY`.

```shell
cd VisionRecBench
source scripts/setup_env.sh
$ISAACSIM_ROOT/python.sh -m pip install openai==1.79.0
```

`agent.py` also accepts the legacy variables `API_KEY` and `BASE_URL`.

## Single Scenarios

`--scenario` selects the experimental condition. `--level` selects the prompt difficulty level from 0 to 3.

Experiment level 1, direct self arm:

```shell
cd VisionRecBench
$ISAACSIM_ROOT/python.sh scripts/inference.py \
  --scenario level1_single_direct \
  --level 1 \
  --model gpt-4o \
  --headless
```

Experiment level 1, random non-self arm:

```shell
$ISAACSIM_ROOT/python.sh scripts/inference.py \
  --scenario level1_single_random \
  --level 1 \
  --model gpt-4o \
  --headless
```

Experiment level 2, fixed scrambled action-space mapping:

```shell
$ISAACSIM_ROOT/python.sh scripts/inference.py \
  --scenario level2_single_scrambled_fixed \
  --level 1 \
  --model gpt-4o \
  --headless
```

Experiment level 3, original three-arm task:

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
  --scenario level1_single_direct \
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

### Inference Options

`scripts/inference.py` accepts these runtime options:

- `--scenario`: scenario name from `tasks/scenario_repo.json`, default `triad_delay_invert`.
- `--arm`: optional arm definition override from `tasks/arm_repo.json`.
- `--level`: prompt difficulty level, one of `0`, `1`, `2`, or `3`, default `1`. This is separate from the scenario's `experiment_level`.
- `--model`: model name to evaluate, or `random` for the random baseline. This option is required.
- `--max_steps`: maximum episode steps. Use `-1` for the scenario default; default `-1`.
- `--max_image_history`: number of previous observations included in the prompt, default `4`.
- `--target_index`: optional 1-based target candidate index override.
- `--seed`: optional scenario random seed override.
- `--headless`: run Isaac Sim in headless mode.

Render quality settings such as renderer, resolution, sampling, and denoising are fixed in `source/render_config.py` so runs use consistent observations.

Isaac Sim may spend one or two minutes after `app ready` compiling shaders and initializing render buffers, especially on the first run in a fresh environment. The many `omni.isaac.* has been deprecated` warnings are emitted by Isaac Sim extensions and are not VisionRecBench errors. A healthy run eventually prints per-step progress and writes outputs to `logs/` and `results/`.

## Full Evaluation

```shell
cd VisionRecBench
chmod +x scripts/evaluate.sh
./scripts/evaluate.sh gpt-4o 1
```

Each run writes observations and logs to `logs/<timestamp>/` and metrics to `results/experiment_level*/prompt_level*/<model>/<scenario>/`.

## Metrics

The result JSON reports:

- `accuracy`: fraction of episode steps where the model selected the correct answer option,
- `experiment_level`: scenario-level task family, independent of prompt difficulty,
- `task_mode`: `single_binary` or `multi_arm`,
- `answer_index` and `answer_options`: the answer option used for scoring,
- `target_present`: for single-arm binary tasks, whether the visible arm is truly self,
- `final_correct`: whether the final answer was correct,
- `majority_correct`: whether the majority selected option was correct,
- `first_correct_step`: first step where the correct answer was selected,
- `bad_response`: unparsable or out-of-range answers.

The core score is `accuracy`; `final_correct` and `majority_correct` are useful when treating an episode as a single identification problem.
