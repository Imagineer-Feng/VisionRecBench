import copy
from itertools import permutations

import numpy as np


def select_behavior_option(options, seed, strategy="random", rng=None):
    if not options:
        raise ValueError("At least one behavior option is required.")

    if strategy == "seed_modulo":
        return int(seed) % len(options)
    if strategy == "random":
        if rng is None:
            rng = np.random.default_rng(int(seed))
        return int(rng.integers(0, len(options)))
    raise ValueError(f"Unsupported behavior selection strategy: {strategy}")


def materialize_mapping_behavior(behavior, seed, behavior_option_count=1):
    resolved = copy.deepcopy(behavior)
    mapping_options = resolved.pop("mapping_options", None)
    if not mapping_options:
        return resolved

    mapping_count = len(mapping_options)
    mapping_offset = (int(seed) // max(1, int(behavior_option_count))) % mapping_count
    behavior_name = resolved.get("behavior")

    if behavior_name == "mapped_direct":
        resolved["mapping"] = copy.deepcopy(mapping_options[mapping_offset])
        resolved["sampled_mapping_option"] = mapping_offset + 1
        return resolved

    if behavior_name == "mapped_cycle_switch":
        mapping_pattern = resolved.pop(
            "mapping_pattern",
            list(range(mapping_count)),
        )
        if not mapping_pattern:
            raise ValueError("mapped_cycle_switch requires a mapping pattern.")
        if any(
            not isinstance(index, int) or not 0 <= index < mapping_count
            for index in mapping_pattern
        ):
            raise ValueError(
                "mapped_cycle_switch mapping_pattern contains an invalid index."
            )
        resolved["mappings"] = [
            copy.deepcopy(mapping_options[(mapping_offset + index) % mapping_count])
            for index in mapping_pattern
        ]
        resolved["mapping_pattern"] = list(mapping_pattern)
        resolved["sampled_mapping_option"] = mapping_offset + 1
        return resolved

    raise ValueError(
        "mapping_options are only supported for mapped_direct and "
        "mapped_cycle_switch behaviors."
    )


def configure_binary_answers(answer_options, target_present, seed, shuffle=False):
    if len(answer_options) != 2:
        raise ValueError("Binary tasks must define exactly two answer options.")

    ordered = list(answer_options)
    semantic_answer_index = 0 if target_present else 1
    if shuffle and (int(seed) // 2) % 2 == 1:
        ordered.reverse()
        semantic_answer_index = 1 - semantic_answer_index

    return ordered, semantic_answer_index + 1


def build_mismatched_command_schedule(
    command_library,
    episode_steps,
    seed,
    mismatch_count,
):
    commands = np.asarray(command_library, dtype=float)
    if commands.ndim != 2 or len(commands) < 2:
        raise ValueError(
            "A mismatched command schedule requires at least two command vectors."
        )
    if len({tuple(command) for command in commands}) != len(commands):
        raise ValueError(
            "A mismatched command schedule requires unique command vectors."
        )
    episode_steps = int(episode_steps)
    mismatch_count = int(mismatch_count)
    if episode_steps != len(commands):
        raise ValueError(
            "Exact mismatch scheduling requires episode_steps to equal the "
            "number of command vectors."
        )
    if not 2 <= mismatch_count <= episode_steps:
        raise ValueError(
            "mismatch_count must be between 2 and episode_steps inclusive."
        )

    rng = np.random.default_rng(int(seed))
    base_indices = np.arange(len(commands))
    mismatch_indices = np.sort(
        rng.choice(base_indices, size=mismatch_count, replace=False)
    )
    mismatch_permutation = None
    for _ in range(1000):
        candidate = rng.permutation(mismatch_indices)
        if np.all(candidate != mismatch_indices):
            mismatch_permutation = candidate
            break
    if mismatch_permutation is None:
        raise RuntimeError("Could not construct an exact mismatch schedule.")

    permutation = base_indices.copy()
    permutation[mismatch_indices] = mismatch_permutation
    schedule = [commands[index].copy() for index in permutation]
    observed_mismatches = 0
    for step, applied in enumerate(schedule):
        expected = commands[step]
        if not np.array_equal(applied, expected):
            observed_mismatches += 1
    if observed_mismatches != mismatch_count:
        raise RuntimeError(
            "Mismatch schedule does not contain the requested number of mismatches."
        )
    return schedule


def build_deranged_command_schedule(command_library, episode_steps, seed):
    return build_mismatched_command_schedule(
        command_library,
        episode_steps=episode_steps,
        seed=seed,
        mismatch_count=episode_steps,
    )


def build_multi_arm_role_assignment(
    num_arms,
    distractors,
    seed,
    target_index=None,
    target_behavior=None,
):
    num_arms = int(num_arms)
    if num_arms < 2:
        raise ValueError("A multi-arm assignment requires at least two arms.")
    if len(distractors) != num_arms - 1:
        raise ValueError(
            f"Expected {num_arms - 1} distractors for {num_arms} arms, "
            f"got {len(distractors)}."
        )

    seed = int(seed)
    if target_index is None:
        target_zero_index = seed % num_arms
    else:
        target_zero_index = int(target_index) - 1
        if not 0 <= target_zero_index < num_arms:
            raise ValueError("target_index must be within [1, num_arms].")

    remaining_positions = [
        index for index in range(num_arms) if index != target_zero_index
    ]
    distractor_orders = list(permutations(range(len(distractors))))
    order_index = (seed // num_arms) % len(distractor_orders)
    distractor_order = distractor_orders[order_index]

    if target_behavior is None:
        target_behavior = {
            "behavior": "direct",
            "desc": "the target arm that follows the motor command directly",
        }
    target_behavior = materialize_mapping_behavior(
        target_behavior,
        seed=seed,
        behavior_option_count=num_arms,
    )
    assignments = [None] * num_arms
    assignments[target_zero_index] = {
        "index": target_zero_index + 1,
        "role": "target",
        "behavior": copy.deepcopy(target_behavior),
    }

    for position, distractor_index in zip(remaining_positions, distractor_order):
        assignments[position] = {
            "index": position + 1,
            "role": "distractor",
            "behavior": materialize_mapping_behavior(
                distractors[distractor_index],
                seed=seed,
                behavior_option_count=num_arms,
            ),
        }

    return assignments


def apply_mapped_behavior(behavior, target_delta, command_index, command_dim):
    behavior_name = behavior["behavior"]
    if behavior_name == "mapped_direct":
        mapping = np.asarray(behavior["mapping"], dtype=float)
    elif behavior_name == "mapped_cycle_switch":
        mappings = behavior["mappings"]
        if not mappings:
            raise ValueError("mapped_cycle_switch requires at least one mapping.")
        cycle_length = int(behavior["cycle_length"])
        if cycle_length < 1:
            raise ValueError("mapped_cycle_switch cycle_length must be positive.")
        cycle_index = int(command_index) // cycle_length
        mapping = np.asarray(mappings[cycle_index % len(mappings)], dtype=float)
    else:
        raise ValueError(f"Unsupported mapped behavior: {behavior_name}")

    expected_shape = (command_dim, command_dim)
    if mapping.shape != expected_shape:
        raise ValueError(
            f"{behavior_name} behavior requires a "
            f"{command_dim}x{command_dim} mapping matrix."
        )
    return mapping @ np.asarray(target_delta, dtype=float)
