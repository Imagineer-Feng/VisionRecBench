import argparse
import copy
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from scripts.inference import (  # noqa: E402
    annotate_candidates,
    get_control_labels,
    make_candidate_motion_panel,
    make_motion_diff,
    save_rgb,
)
from source.dataset_io import (  # noqa: E402
    DATASET_SCHEMA_VERSION,
    atomic_write_json,
    dataset_content_hash,
    image_descriptor,
    load_json,
    rebuild_manifest,
    utc_timestamp,
    validate_dataset,
)
from source.episode_sampling import (  # noqa: E402
    SAMPLING_PROFILE,
    STANDARD_SCENARIOS,
    build_episode_task,
)
from source.render_config import RENDER_CONFIG  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a frozen, model-independent VisionRecBench dataset. "
            "Run this script with Isaac Sim's python.sh."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "datasets" / "visionrecbench_robust_v2",
    )
    parser.add_argument(
        "--dataset-name",
        default="visionrecbench_robust_v2",
    )
    parser.add_argument(
        "--scenario",
        dest="scenarios",
        nargs="+",
        choices=STANDARD_SCENARIOS,
        default=list(STANDARD_SCENARIOS),
    )
    parser.add_argument("--episodes-per-scene", type=int, default=48)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--profile", default=SAMPLING_PROFILE)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already completed episodes after checking their signatures.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate the deterministic sampling plan without starting Isaac Sim.",
    )
    parser.add_argument(
        "--skip-checksums",
        action="store_true",
        help="Skip the final full-file checksum pass (generation still stores checksums).",
    )
    args = parser.parse_args()
    if args.episodes_per_scene < 1:
        parser.error("--episodes-per-scene must be positive")
    if args.episodes_per_scene % 12 != 0:
        parser.error(
            "--episodes-per-scene must be divisible by 12 so conditions, "
            "answer positions, mappings, and candidate roles are balanced"
        )
    if args.base_seed < 0:
        parser.error("--base-seed must be non-negative")
    return args


def sampling_plan(args):
    tasks = []
    for scenario in args.scenarios:
        for episode_index in range(args.episodes_per_scene):
            tasks.append(
                build_episode_task(
                    scenario,
                    episode_index,
                    base_seed=args.base_seed,
                    profile=args.profile,
                )
            )
    signatures = [task["episode_signature"] for task in tasks]
    duplicate_count = len(signatures) - len(set(signatures))
    if duplicate_count:
        raise RuntimeError(
            f"Sampling plan contains {duplicate_count} duplicate episode signatures."
        )
    return tasks


def plan_report(tasks):
    scenario_counts = Counter(task["name"] for task in tasks)
    return {
        "profile": SAMPLING_PROFILE,
        "episode_count": len(tasks),
        "scenario_counts": dict(scenario_counts),
        "unique_signatures": len(
            {task["episode_signature"] for task in tasks}
        ),
        "first_episode_ids": [task["episode_id"] for task in tasks[:3]],
    }


def metadata_for(args):
    try:
        source_revision = subprocess.check_output(
            ["git", "-C", str(BASE_DIR), "describe", "--always", "--dirty"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        source_revision = "unknown"
    source_digest = hashlib.sha256()
    source_paths = (
        BASE_DIR / "scripts" / "generate_dataset.py",
        BASE_DIR / "scripts" / "inference.py",
        BASE_DIR / "source" / "dataset_io.py",
        BASE_DIR / "source" / "episode_sampling.py",
        BASE_DIR / "source" / "env.py",
        BASE_DIR / "source" / "task_logic.py",
        BASE_DIR / "source" / "render_config.py",
        BASE_DIR / "tasks" / "scenario_repo.json",
        BASE_DIR / "tasks" / "arm_repo.json",
        BASE_DIR / "tasks" / "distractor_repo.json",
    )
    for source_path in source_paths:
        source_digest.update(source_path.relative_to(BASE_DIR).as_posix().encode())
        source_digest.update(source_path.read_bytes())
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_name": args.dataset_name,
        "sampling_profile": args.profile,
        "scenarios": list(args.scenarios),
        "episodes_per_scene": int(args.episodes_per_scene),
        "base_seed": int(args.base_seed),
        "render_config": copy.deepcopy(RENDER_CONFIG),
        "source_revision": source_revision,
        "generator_source_sha256": source_digest.hexdigest(),
        "created_at": utc_timestamp(),
        "completed_episodes": 0,
    }


def prepare_dataset_root(args):
    output = args.output.resolve()
    metadata_path = output / "metadata.json"
    expected = metadata_for(args)
    if output.exists() and any(output.iterdir()):
        if not args.resume:
            raise FileExistsError(
                f"Dataset output is not empty: {output}. Use --resume to continue."
            )
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Cannot resume without dataset metadata: {metadata_path}"
            )
        existing = load_json(metadata_path)
        keys = (
            "schema_version",
            "dataset_name",
            "sampling_profile",
            "scenarios",
            "episodes_per_scene",
            "base_seed",
            "render_config",
            "generator_source_sha256",
        )
        mismatches = [key for key in keys if existing.get(key) != expected.get(key)]
        if mismatches:
            raise ValueError(
                "Resume configuration differs from metadata for: "
                + ", ".join(mismatches)
            )
        return output, existing

    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(metadata_path, expected)
    return output, expected


def apply_render_config(task):
    task["anti_aliasing_op"] = RENDER_CONFIG["anti_aliasing"]
    task["pathtracing_spp"] = RENDER_CONFIG["pathtracing_spp"]
    task["denoiser_enabled"] = RENDER_CONFIG["denoiser"]
    return task


def render_episode(env, task, dataset_root):
    episode_id = task["episode_id"]
    episode_dir = dataset_root / "episodes" / episode_id
    image_dir = episode_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    should_annotate = bool(task.get("annotate_candidates", True))
    num_arms = int(task["num_arms"])
    evidence_mode = task.get("visual_history_mode", "observations")
    control_labels = get_control_labels(task)

    initial = env.reset()
    if should_annotate:
        initial = annotate_candidates(initial, num_arms)
    initial_path = image_dir / "step_000_initial.png"
    initial = save_rgb(initial, initial_path)

    previous = initial
    steps = []
    for step_index in range(1, int(task["episode_steps"]) + 1):
        command = env.get_command(step_index)
        observation, applied_commands = env.step(command)
        if should_annotate:
            observation = annotate_candidates(observation, num_arms)
        observation_path = image_dir / f"step_{step_index:03d}_observation.png"
        observation = save_rgb(observation, observation_path)

        if evidence_mode == "candidate_motion_panels":
            evidence = make_candidate_motion_panel(
                previous,
                observation,
                num_arms,
                annotate=should_annotate,
            )
        elif evidence_mode == "motion_diffs":
            evidence = make_motion_diff(
                previous,
                observation,
                num_arms,
                annotate=should_annotate,
            )
        else:
            evidence = observation
        evidence_path = image_dir / f"step_{step_index:03d}_evidence.png"
        save_rgb(evidence, evidence_path)

        steps.append(
            {
                "step": step_index,
                "command": command,
                "applied_commands": applied_commands,
                "observation": image_descriptor(dataset_root, observation_path),
                "evidence": image_descriptor(dataset_root, evidence_path),
            }
        )
        previous = observation

    resolved_task = copy.deepcopy(env.task_dict)
    record = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "episode_id": episode_id,
        "episode_index": int(task["episode_variation"]["episode_index"]),
        "episode_signature": task["episode_signature"],
        "scenario": task["name"],
        "scene": int(task["scene"]),
        "seed": int(task["seed"]),
        "sampling_profile": task["episode_variation"]["profile"],
        "task": resolved_task,
        "control_labels": control_labels,
        "visual_history_mode": evidence_mode,
        "target_present": bool(env.target_present),
        "target_index": env.target_index,
        "answer_index": int(env.answer_index),
        "answer_text": env.answer_options[env.answer_index - 1],
        "answer_options": list(env.answer_options),
        "candidates": copy.deepcopy(env.candidates),
        "initial_observation": image_descriptor(dataset_root, initial_path),
        "steps": steps,
    }
    atomic_write_json(episode_dir / "episode.json", record)
    return record


def update_progress_metadata(dataset_root, metadata):
    rows = rebuild_manifest(dataset_root)
    metadata["completed_episodes"] = len(rows)
    metadata["content_sha256"] = dataset_content_hash(rows)
    metadata["updated_at"] = utc_timestamp()
    atomic_write_json(dataset_root / "metadata.json", metadata)
    return rows


def generate(args, tasks):
    dataset_root, metadata = prepare_dataset_root(args)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": args.headless,
            "renderer": RENDER_CONFIG["renderer"],
            "width": RENDER_CONFIG["resolution"],
            "height": RENDER_CONFIG["resolution"],
            "anti_aliasing": RENDER_CONFIG["anti_aliasing"],
            "denoiser": RENDER_CONFIG["denoiser"],
        }
    )
    from source.env import VisionRecBenchEnv

    try:
        for completed_index, sampled_task in enumerate(tasks, start=1):
            record_path = (
                dataset_root
                / "episodes"
                / sampled_task["episode_id"]
                / "episode.json"
            )
            if record_path.is_file():
                existing = load_json(record_path)
                if (
                    existing.get("episode_signature")
                    != sampled_task["episode_signature"]
                ):
                    raise ValueError(
                        f"Existing episode signature mismatch: {record_path}"
                    )
                print(
                    f"[{completed_index}/{len(tasks)}] skipping "
                    f"{sampled_task['episode_id']}",
                    flush=True,
                )
                continue

            print(
                f"[{completed_index}/{len(tasks)}] rendering "
                f"{sampled_task['episode_id']}",
                flush=True,
            )
            task = apply_render_config(copy.deepcopy(sampled_task))
            env = None
            try:
                env = VisionRecBenchEnv(simulation_app, task)
                render_episode(env, task, dataset_root)
            finally:
                if env is not None:
                    env.close(close_simulation_app=False)
            update_progress_metadata(dataset_root, metadata)
    finally:
        simulation_app.close()

    update_progress_metadata(dataset_root, metadata)
    report = validate_dataset(
        dataset_root,
        verify_checksums=not args.skip_checksums,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["valid"]:
        raise SystemExit(1)


def main():
    args = parse_args()
    tasks = sampling_plan(args)
    if args.plan_only:
        print(json.dumps(plan_report(tasks), indent=2, sort_keys=True))
        return
    generate(args, tasks)


if __name__ == "__main__":
    main()
