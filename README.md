# VisionRecBench: Visual Self-Recognition with Mimic Robot Arms

VisionRecBench is a standalone Isaac Sim benchmark for embodied self-recognition. The standard benchmark scenarios use a Franka Panda robotic arm loaded from the Isaac Sim asset library instead of the earlier simple procedural arm. It now supports three scenes:

- Scene 1: one visible arm, binary same-step command-causality judgment. The self arm follows the current command directly; the non-self arm executes the same command set in a deranged order with no accidental same-step matches.
- Scene 2: one visible arm under a scrambled action space. A four-command cycle repeats three times; the hidden mapping is either stable across all cycles (self) or changes between cycles (non-self).
- Scene 3: a balanced two-arm visual self-recognition task. One candidate is the target agent's own arm and follows the current command; the other follows the previous command with a one-step delay. Self and delay roles are balanced across left-to-right positions.

The agent receives:

- the current RGB observation,
- a short visual history or per-candidate motion-panel history,
- the motor-command history available to the agent,
- and either binary self/non-self answer options or candidate arm IDs ordered from left to right.

The task is to answer whether the visible arm is itself, or which visible candidate arm is itself, depending on the scenario.

## Project Layout

```text
VisionRecBench/
  source/
    action.py               # candidate answer schema
    agent.py                # OpenAI-compatible and random candidate identifiers
    env.py                  # Isaac Sim environment and Panda/USD loading
    preprocess.py           # scenario/config loading
    prompts.py              # prompt difficulty levels 0-3
    render_config.py        # fixed renderer, resolution, and sampling settings
    task_logic.py           # balanced sampling and command/mapping transformations
  tasks/
    arm_repo.json           # arm definitions; standard scenarios use panda_arm
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

Scene 1, balanced command-causality benchmark:

```shell
cd VisionRecBench
$ISAACSIM_ROOT/python.sh scripts/inference.py \
  --scenario scene1_single_command_causality \
  --level 1 \
  --model gpt-4o \
  --seed 0 \
  --headless
```

For Scene 1, consecutive seeds alternate direct self and deranged-order
non-self conditions. Both conditions use exactly the same command multiset and
therefore the same total action budget; only same-step correspondence differs.
Answer positions are balanced over every four consecutive seeds. The model
receives the complete per-step motion-difference trace and makes one scored
judgment after all eight actions. The legacy name
`scene1_single_direct_or_random` remains accepted as an alias.

The fixed `scene1_single_direct` and `scene1_single_deranged` scenarios remain
available as diagnostic conditions:

```shell
$ISAACSIM_ROOT/python.sh scripts/inference.py \
  --scenario scene1_single_direct \
  --level 1 \
  --model gpt-4o \
  --headless
```

The legacy diagnostic name `scene1_single_random` remains accepted as an alias
for `scene1_single_deranged`.

Scene 2, balanced stable-versus-changing scrambled mapping:

```shell
$ISAACSIM_ROOT/python.sh scripts/inference.py \
  --scenario scene2_single_scrambled_stability \
  --level 1 \
  --model gpt-4o \
  --seed 0 \
  --headless
```

For Scene 2, consecutive seeds alternate self/non-self conditions. The answer
option order also changes in a balanced four-seed pattern. Seed `0` is a stable
self case and seed `1` is a changing-mapping non-self case. The legacy scenario
name `scene2_single_scrambled_fixed` remains accepted as an alias. The model
receives the complete sequence of per-action motion-difference images and makes
one scored judgment after all three cycles. Single-arm Scene 2 images omit the
otherwise-unnecessary candidate number marker to avoid an Option 1 visual bias.

Scene 3, balanced two-arm causal-identification task:

```shell
cd VisionRecBench
$ISAACSIM_ROOT/python.sh scripts/inference.py \
  --scenario scene3_dyad_causal_identification \
  --level 1 \
  --model gpt-4o \
  --seed 0 \
  --headless
```

For Scene 3, one candidate follows the current command directly and the other
follows the previous step's command. The delayed candidate remains still on the
first step because no previous command exists. Consecutive seeds alternate the
self and delay roles across the two left-to-right positions, so two consecutive
seeds cover both target positions. The old names
`scene3_triad_causal_identification` and `scene3_triad_delay_invert` remain
accepted as compatibility aliases. The model receives per-candidate motion
panels for every step: each panel contains before, after, and signed-change crops
for both candidates. Scene 3 makes one scored judgment after the full
eight-action diagnostic sequence.

Random baseline:

```shell
$ISAACSIM_ROOT/python.sh scripts/inference.py \
  --scenario scene1_single_direct \
  --level 1 \
  --model random \
  --headless
```

The default run uses `PathTracing`, 1024x1024 observations, 16 samples per pixel, and enough render frames for the camera buffer to settle. This is intentionally slower than the old fast defaults because it avoids the noisy, blurry `RayTracedLighting` output.

### Inference Options

`scripts/inference.py` accepts these runtime options:

- `--scenario`: scenario name from `tasks/scenario_repo.json`, default `scene3_dyad_causal_identification`.
- `--arm`: advanced arm definition override from `tasks/arm_repo.json`. Standard benchmark scenarios are configured to use `panda_arm`.
- `--level`: prompt difficulty level, one of `0`, `1`, `2`, or `3`, default `1`. This is separate from the scenario's `scene`.
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
./scripts/evaluate.sh gpt-4o 1 12
```

Each run writes observations and logs to `logs/<timestamp>/` and metrics to `results/scene*/prompt_level*/<model>/<scenario>/`.
The third argument is the number of runs per scenario, not the total number of
runs. It must be divisible by twelve so Scenes 1 and 2 have balanced binary
labels/answer positions and Scene 3 has balanced target/role positions.

## Frozen Offline Dataset Workflow

For paper experiments, prefer the frozen dataset workflow over repeatedly
rendering the same scene for every model call. It separates Isaac Sim episode
generation from API evaluation, guarantees that every model receives identical
images, and treats an episode rather than an API call as the experimental item.

The default dataset name is `visionrecbench_robust_v2` because Scene 3 changed
from the legacy three-arm design to the two-arm self-versus-delay design. Existing
`visionrecbench_robust_v1` files remain immutable legacy data and should not be
mixed with newly generated episodes.

First inspect the deterministic 144-episode sampling plan without launching
Isaac Sim:

```shell
python3 scripts/generate_dataset.py --plan-only
```

Generate 48 independent episodes per standard scene. The count must be
divisible by 12 so labels, answer positions, Scene 2 mappings, and Scene 3 role
positions remain balanced:

```shell
$ISAACSIM_ROOT/python.sh scripts/generate_dataset.py \
  --output datasets/visionrecbench_robust_v2 \
  --episodes-per-scene 48 \
  --base-seed 0 \
  --headless
```

The `robust_v1` profile deterministically varies command order and amplitude,
initial joint pose, camera pose and focal length, lighting, floor color, and
background color. Generation writes one immutable `episode.json` per episode,
plus a dataset-wide `manifest.jsonl` and `metadata.json`. Every model-facing
image stores its SHA-256 checksum and dimensions. Interrupted generation can be
continued with `--resume`; completed episode signatures are checked before they
are skipped.

Validate structure, image readability, checksums, uniqueness, and balance:

```shell
python3 scripts/validate_dataset.py datasets/visionrecbench_robust_v2
```

Evaluate a model from the frozen files without starting Isaac Sim:

```shell
python3 scripts/evaluate_dataset.py \
  --dataset datasets/visionrecbench_robust_v2 \
  --model gpt-4o \
  --level 1 \
  --resume
```

Pass multiple prompt levels, for example `--level 0 1 2 3`, when running a
prompt ablation. Offline results are written under
`results/offline/<dataset>/<prompt-level>/<model>/<scenario>/`. Each result
records the frozen dataset hash, exact multimodal input hash, raw response,
parsed choice, label, and correctness. Use `--model random` for an API-free
pipeline smoke test.

Summarize completed results using independent episodes as the bootstrap unit:

```shell
python3 scripts/summarize_offline_results.py \
  results/offline/visionrecbench_robust_v2 \
  --output results/offline/visionrecbench_robust_v2/summary.json
```

The summary reports accuracy and invalid-response rate for every group,
self/non-self recall, balanced accuracy and self-attribution rate for binary
scenes, and per-position recall and prediction distributions for multi-arm
scenes. Confidence intervals are episode-level bootstrap intervals rather than
intervals over repeated API calls.

The legacy `scripts/inference.py` entrypoint remains useful for interactive
single-run debugging, but its repeated runs should not be counted as new
benchmark items unless their physical/visual episode configurations differ.

## Metrics

The result JSON reports:

- `accuracy`: fraction of model judgments where the model selected the correct answer option,
- `prediction_steps`: number of LLM judgments made during the run; rebuilt standard Scenes 1, 2, and 3 each make one episode-level judgment,
- `scene`: scene task family, independent of prompt difficulty,
- `task_mode`: `single_binary` or `multi_arm`,
- `answer_index` and `answer_options`: the answer option used for scoring,
- `target_present`: for single-arm binary tasks, whether the visible arm is truly self,
- `final_correct`: whether the final answer was correct,
- `majority_correct`: whether the majority selected option was correct,
- `first_correct_step`: first step where the correct answer was selected,
- `bad_response`: unparsable or out-of-range answers.
- `action_trace`: every commanded and actually applied action, including steps that were not individually scored.

The core score is `accuracy`; `final_correct` and `majority_correct` are useful when treating an episode as a single identification problem.
