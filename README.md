# VisionRecBench: Visual Self-Recognition with Mimic Robot Arms

VisionRecBench is an Isaac Sim benchmark for embodied visual self-recognition. It renders frozen robot-arm episodes once, then evaluates every model offline from the same images and motor-command traces. The benchmark no longer provides an online render-and-evaluate path.

The three standard scenes are:

- **Scene 1 — same-step command causality:** one arm is shown. A self arm follows the current command; a non-self arm executes a reordered version of the same eight-command multiset.
- **Scene 2 — mapping stability:** one arm is shown under a scrambled action space. Four command cycles are observed; a self arm keeps one command-to-motion mapping, while a non-self arm changes mappings between cycles according to the selected difficulty.
- **Scene 3 — delayed-arm identification:** two arms are shown. One is the self arm and follows the current command; the other executes the same command stream after a difficulty-dependent delay. Self/delayed roles are balanced across left and right positions.

Each episode produces one scored answer after the complete motion trace.

## Difficulty levels

`--level` now selects **physical task difficulty**, not prompt detail. The same prompt template is used for every scene and level.

| Level | Scene 1: wrong same-step actions | Scene 2: non-self cycle mappings | Scene 3: delayed arm |
|---|---:|---|---:|
| 1 — easy | 8 of 8 | `A, B, C, D` | 3 steps |
| 2 — medium | 6 of 8 | `A, A, B, C` | 2 steps |
| 3 — hard | 4 of 8 | `A, A, A, B` | 1 step |

Here `A`–`D` denote mutually different deranged command-to-motion mappings. Self episodes in Scenes 1 and 2 retain direct or stable behavior, respectively; the difficulty gradient changes the non-self comparison condition.

The universal prompt asks the model to use the complete temporal evidence and does not name mismatch counts, mapping repetition patterns, delay lengths, difficulty names, or the correct option.

## Project layout

```text
VisionRecBench/
  source/
    action.py               # answer-option schema
    agent.py                # offline model and random-baseline adapters
    dataset_io.py           # frozen dataset records and validation
    difficulty.py           # level names and per-level config resolution
    env.py                  # Isaac Sim environment and Panda loading
    environment_config.py   # balanced procedural laboratory templates
    episode_sampling.py     # balanced deterministic episode sampling
    multimodal.py           # shared image evidence and model input assembly
    preprocess.py           # scenario/config loading
    prompts.py              # single universal prompt template
    render_config.py        # fixed rendering settings
    task_logic.py           # command and mapping transformations
  tasks/
    arm_repo.json
    distractor_repo.json
    scenario_repo.json
  scripts/
    generate_dataset.py
    render_environment_previews.py
    validate_dataset.py
    evaluate_dataset.py
    summarize_offline_results.py
    setup_env.example.sh
  tests/
```

## Setup

Dataset generation requires Isaac Sim and `ISAACSIM_ROOT`. API-backed offline evaluation additionally requires the Python `openai` package and `OPENAI_API_KEY`.

```shell
cd VisionRecBench
cp scripts/setup_env.example.sh scripts/setup_env.sh
# Edit local paths and credentials, then:
source scripts/setup_env.sh
$ISAACSIM_ROOT/python.sh -m pip install openai==1.79.0
```

`source/agent.py` also accepts the legacy `API_KEY` and `BASE_URL` environment variables.

## 1. Inspect the deterministic plan

This check does not launch Isaac Sim:

```shell
python3 scripts/generate_dataset.py --plan-only
```

With the defaults it plans 432 unique episodes: 48 episodes × 3 scenes × 3 difficulty levels. `--episodes-per-scene` is the count **per scene per level** and must be divisible by 12 to preserve condition, answer-position, mapping, and candidate-role balance.

To inspect a subset:

```shell
python3 scripts/generate_dataset.py \
  --plan-only \
  --scenario scene1_single_command_causality \
  --level 1 2 3 \
  --episodes-per-scene 12
```

## 2. Generate the frozen dataset

Run generation with Isaac Sim's Python:

```shell
$ISAACSIM_ROOT/python.sh scripts/generate_dataset.py \
  --output datasets/visionrecbench_robust_v3 \
  --dataset-name visionrecbench_robust_v3 \
  --episodes-per-scene 48 \
  --level 1 2 3 \
  --base-seed 0 \
  --headless
```

The `paired_lab_v3` sampler uses strict two-episode nuisance pairs. For every scene, indices `2k` and `2k+1` share the exact command order and amplitude, initial pose, camera, lighting, floor, background, and procedural laboratory environment. Scene 1/2 change only between self and non-self behavior; Scene 3 changes only which left/right candidate is self. The same pair is also reused at all selected difficulty levels.

The visual context rotates evenly among three procedural templates: `robotics_lab`, `assembly_cell`, and `inspection_bay`. They add walls, floor markings, structural beams, workbenches, monitors, cabinets, shelving, bins, safety fencing, crates, calibration panels, and inspection consoles without placing answer-correlated objects near a candidate. These first-version props are static visual geometry with no collision or independent motion, keeping background richness separate from causal-task difficulty. Because every valid episode count is divisible by 12, every template appears equally often in every scene, difficulty, and answer condition.

Render one initial-observation preview for each template before generating the full dataset:

```shell
$ISAACSIM_ROOT/python.sh scripts/render_environment_previews.py --headless
```

Each record stores a `nuisance_pair_id`, a content-derived `nuisance_signature`, and its `environment_template`. The deterministic plan and dataset validator reject incomplete pairs, unequal signatures, unbalanced templates, or pair sets that differ across levels. With the default plan there are 72 nuisance pairs, each reused by two conditions across three levels, giving six episodes per pair. Generation also writes image checksums, `manifest.jsonl`, and `metadata.json`. Use `--resume` to continue an interrupted generation with matching configuration and source hashes.

## 3. Validate the dataset

```shell
python3 scripts/validate_dataset.py datasets/visionrecbench_robust_v3
```

Validation checks record structure, images, checksums, episode uniqueness, per-scene/per-level balance, equal environment-template frequency, and strict nuisance-pair equality across conditions and levels. `--skip-checksums` provides a faster structural check.

## 4. Evaluate offline

No Isaac Sim process is started during evaluation:

```shell
python3 scripts/evaluate_dataset.py \
  --dataset datasets/visionrecbench_robust_v3 \
  --model gpt-4o \
  --level 1 2 3 \
  --resume
```

Omit `--level` to evaluate all physical difficulty levels. Use `--scenario` to select scenes, `--limit` for a smoke test, or `--model random` for an API-free baseline.

Results are written to:

```text
results/offline/<dataset>/difficulty_level<level>/<model>/<scenario>/<episode>.json
```

Each result records the frozen dataset hash, exact multimodal-input hash, episode signature, nuisance-pair identity, difficulty, raw response, parsed choice, label, and correctness. The evaluator requires both the new difficulty-level schema and strict nuisance-pair metadata; legacy or unpaired datasets remain readable by the validator but are not silently mixed into new experiments.

## 5. Summarize results

```shell
python3 scripts/summarize_offline_results.py \
  results/offline/visionrecbench_robust_v3 \
  --output results/offline/visionrecbench_robust_v3/summary.json
```

The summary reports accuracy and invalid-response rate per model, scene, and difficulty. Binary scenes also report self/non-self recall, balanced accuracy, and self-attribution rate; the two-arm scene reports position recall and prediction distributions. When complete paired results are available, confidence intervals use nuisance-pair cluster bootstrap rather than incorrectly treating the two matched episodes as independent.

## Reproducibility and legacy data

- Render quality is fixed in `source/render_config.py`.
- Prompt text is fixed in `source/prompts.py` and is shared across all nine scene/level combinations.
- Existing legacy datasets and results are not modified. In particular, `visionrecbench_robust_v1` represents the old experimental design and should not be combined with `visionrecbench_robust_v3`.
- A change to scene physics, sampling, evidence construction, or prompts should produce a newly named frozen dataset rather than overwriting an existing one.
