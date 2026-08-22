import copy
import hashlib
import json

import numpy as np

from source.difficulty import DIFFICULTY_LEVELS
from source.environment_config import sample_environment
from source.preprocess import construct
from source.task_logic import (
    build_mismatched_command_schedule,
    build_multi_arm_role_assignment,
    materialize_mapping_behavior,
    select_behavior_option,
)


STANDARD_SCENARIOS = (
    "scene1_single_command_causality",
    "scene2_single_scrambled_stability",
    "scene3_dyad_causal_identification",
)

SAMPLING_PROFILE = "paired_lab_v3"
NUISANCE_PAIR_SIZE = 2


def _round_list(values, digits=6):
    return [round(float(value), digits) for value in values]


def _episode_rng(base_seed, scene, episode_index):
    # Consecutive condition episodes and all difficulty levels deliberately use
    # the same random stream. Any future nuisance/environment variation must be
    # sampled from this RNG so it remains condition- and level-independent.
    nuisance_pair_index = int(episode_index) // NUISANCE_PAIR_SIZE
    seed_sequence = np.random.SeedSequence(
        [int(base_seed), int(scene), nuisance_pair_index]
    )
    return np.random.default_rng(seed_sequence)


def _stable_hash(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _vary_commands(task, rng):
    commands = copy.deepcopy(task["command_sequence"])
    scene = int(task["scene"])

    # Scene 2 needs one repeated four-command cycle. The other scenes use one
    # full diagnostic sequence, so all commands can be permuted directly.
    order = rng.permutation(len(commands)).tolist()
    varied = []
    amplitudes = []
    for source_index in order:
        command = copy.deepcopy(commands[source_index])
        amplitude = float(rng.uniform(0.78, 1.22))
        delta = np.asarray(command["delta"], dtype=float) * amplitude
        command["delta"] = _round_list(delta)
        varied.append(command)
        amplitudes.append(round(amplitude, 6))

    if scene == 2 and len(varied) != 4:
        raise ValueError("Scene 2 sampling requires a four-command base cycle.")

    task["command_sequence"] = varied
    return {
        "command_source_order": [index + 1 for index in order],
        "command_amplitudes": amplitudes,
    }


def _vary_initial_pose(task, rng):
    arm = task["arm"]
    if "initial_joint_positions" in arm:
        key = "initial_joint_positions"
        limits_key = "joint_limits"
        jitter = 0.10
    else:
        key = "initial_joints_deg"
        limits_key = "joint_limits_deg"
        jitter = 4.0

    initial = np.asarray(arm[key], dtype=float)
    limits = np.asarray(arm[limits_key], dtype=float)
    varied = initial.copy()

    # Panda finger joints have a very small range and should remain fixed.
    movable = np.where((limits[:, 1] - limits[:, 0]) > 0.25)[0]
    varied[movable] += rng.uniform(-jitter, jitter, size=len(movable))
    margin = np.minimum((limits[:, 1] - limits[:, 0]) * 0.08, jitter)
    varied = np.clip(varied, limits[:, 0] + margin, limits[:, 1] - margin)
    arm[key] = _round_list(varied)
    return {
        "initial_pose_key": key,
        "initial_joint_positions": copy.deepcopy(arm[key]),
    }


def _vary_camera(task, rng):
    eye = np.asarray(task["camera_eye"], dtype=float)
    target = np.asarray(task["camera_target"], dtype=float)
    eye += rng.uniform(
        low=np.array([-0.18, -0.16, -0.12]),
        high=np.array([0.18, 0.16, 0.12]),
    )
    target += rng.uniform(
        low=np.array([-0.08, -0.04, -0.06]),
        high=np.array([0.08, 0.04, 0.06]),
    )
    focal_scale = float(rng.uniform(0.96, 1.04))
    task["camera_eye"] = _round_list(eye)
    task["camera_target"] = _round_list(target)
    task["camera_focal"] = round(float(task["camera_focal"]) * focal_scale, 6)
    return {
        "camera_eye": copy.deepcopy(task["camera_eye"]),
        "camera_target": copy.deepcopy(task["camera_target"]),
        "camera_focal": task["camera_focal"],
    }


def _vary_lighting_and_colors(task, rng):
    intensity_keys = (
        "key_light_intensity",
        "dome_light_intensity",
        "fill_light_intensity",
    )
    intensities = {}
    for key in intensity_keys:
        scale = float(rng.uniform(0.82, 1.18))
        task[key] = round(float(task[key]) * scale, 6)
        intensities[key] = task[key]

    rotation = np.asarray(task["key_light_rotation"], dtype=float)
    rotation += rng.uniform(-5.0, 5.0, size=3)
    task["key_light_rotation"] = _round_list(rotation)

    colors = {}
    for key in ("floor_color", "background_color"):
        color = np.asarray(task[key], dtype=float)
        color += rng.uniform(-0.025, 0.025, size=3)
        task[key] = _round_list(np.clip(color, 0.02, 0.98))
        colors[key] = copy.deepcopy(task[key])

    return {
        **intensities,
        "key_light_rotation": copy.deepcopy(task["key_light_rotation"]),
        **colors,
    }


def _condition_descriptor(task):
    if task.get("task_mode") == "single_binary":
        options = task.get("visible_arm_behavior_options")
        if options:
            option_index = select_behavior_option(
                options,
                seed=task["seed"],
                strategy=task.get("behavior_selection", "random"),
            )
            selected = options[option_index]
            behavior = materialize_mapping_behavior(
                selected["behavior"],
                seed=task["seed"],
                behavior_option_count=len(options),
            )
            target_present = bool(selected["target_present"])
        else:
            behavior = materialize_mapping_behavior(
                task["visible_arm_behavior"],
                seed=task["seed"],
            )
            target_present = bool(task.get("target_present", True))

        descriptor = {
            "target_present": target_present,
            "behavior": behavior,
        }
        if behavior["behavior"] in {"sequence_derangement", "sequence_mismatch"}:
            mismatch_count = int(
                behavior.get("mismatch_count", task["episode_steps"])
            )
            schedule = build_mismatched_command_schedule(
                [item["delta"] for item in task["command_sequence"]],
                episode_steps=task["episode_steps"],
                seed=task["seed"],
                mismatch_count=mismatch_count,
            )
            descriptor["applied_schedule"] = [
                _round_list(command) for command in schedule
            ]
        return descriptor

    return {
        "role_assignments": build_multi_arm_role_assignment(
            task["num_arms"],
            task.get("distractors", []),
            seed=task["seed"],
            target_index=task.get("target_index"),
        )
    }


def canonical_episode_signature(task):
    """Return a stable signature of the physical/visual episode design."""
    signature_payload = {
        "scenario": task["name"],
        "difficulty_level": task["difficulty_level"],
        "condition": _condition_descriptor(task),
        "command_sequence": task["command_sequence"],
        "arm_initial_joint_positions": task["arm"].get(
            "initial_joint_positions",
            task["arm"].get("initial_joints_deg"),
        ),
        "camera_eye": task.get("camera_eye"),
        "camera_target": task.get("camera_target"),
        "camera_focal": task.get("camera_focal"),
        "key_light_intensity": task.get("key_light_intensity"),
        "key_light_rotation": task.get("key_light_rotation"),
        "dome_light_intensity": task.get("dome_light_intensity"),
        "fill_light_intensity": task.get("fill_light_intensity"),
        "floor_color": task.get("floor_color"),
        "background_color": task.get("background_color"),
        "environment": task.get("environment"),
    }
    return _stable_hash(signature_payload)


def build_episode_task(
    scenario,
    episode_index,
    level=1,
    base_seed=0,
    profile=SAMPLING_PROFILE,
):
    if profile != SAMPLING_PROFILE:
        raise ValueError(f"Unsupported sampling profile: {profile}")
    if int(episode_index) < 0:
        raise ValueError("episode_index must be non-negative.")
    if int(level) not in DIFFICULTY_LEVELS:
        raise ValueError(f"Unsupported difficulty level: {level}")

    task = construct({"scenario": scenario, "level": int(level)})
    scene = int(task["scene"])
    nuisance_pair_index = int(episode_index) // NUISANCE_PAIR_SIZE
    nuisance_pair_id = (
        f"{task['name']}-base{int(base_seed)}-pair{nuisance_pair_index:05d}"
    )
    # Keep the two condition seeds consecutive with a shared quotient for all
    # non-label sampling based on seed // 2, regardless of base_seed parity.
    episode_seed = 2 * int(base_seed) + int(episode_index)
    rng = _episode_rng(base_seed, scene, episode_index)
    task["seed"] = episode_seed

    environment = sample_environment(
        rng,
        nuisance_pair_index=nuisance_pair_index,
        base_seed=base_seed,
    )
    task["environment"] = copy.deepcopy(environment)
    nuisance_variation = {
        "commands": _vary_commands(task, rng),
        "initial_pose": _vary_initial_pose(task, rng),
        "camera": _vary_camera(task, rng),
        "appearance": _vary_lighting_and_colors(task, rng),
        "environment": environment,
    }
    nuisance_signature = _stable_hash(nuisance_variation)
    variation = {
        "profile": profile,
        "base_seed": int(base_seed),
        "episode_index": int(episode_index),
        "episode_seed": episode_seed,
        "difficulty_level": int(level),
        "nuisance_pair_size": NUISANCE_PAIR_SIZE,
        "nuisance_pair_index": nuisance_pair_index,
        "nuisance_pair_id": nuisance_pair_id,
        "nuisance_signature": nuisance_signature,
        **nuisance_variation,
    }
    task["nuisance_pair_id"] = nuisance_pair_id
    task["nuisance_signature"] = nuisance_signature
    task["episode_variation"] = variation
    task["episode_signature"] = canonical_episode_signature(task)
    task["episode_id"] = (
        f"{task['name']}-level{int(level)}-{int(episode_index):05d}-"
        f"{task['episode_signature'][:12]}"
    )
    return task
