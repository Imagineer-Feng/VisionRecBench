import copy

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
        resolved["mappings"] = [
            copy.deepcopy(mapping_options[(mapping_offset + index) % mapping_count])
            for index in range(mapping_count)
        ]
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


def build_deranged_command_schedule(command_library, episode_steps, seed):
    commands = np.asarray(command_library, dtype=float)
    if commands.ndim != 2 or len(commands) < 2:
        raise ValueError(
            "A deranged command schedule requires at least two command vectors."
        )
    if len({tuple(command) for command in commands}) != len(commands):
        raise ValueError(
            "A deranged command schedule requires unique command vectors."
        )

    rng = np.random.default_rng(int(seed))
    base_indices = np.arange(len(commands))
    permutation = None
    for _ in range(1000):
        candidate = rng.permutation(base_indices)
        if np.all(candidate != base_indices):
            permutation = candidate
            break
    if permutation is None:
        raise RuntimeError("Could not construct a deranged command schedule.")

    schedule = [
        commands[permutation[step % len(commands)]].copy()
        for step in range(int(episode_steps))
    ]
    for step, applied in enumerate(schedule):
        expected = commands[step % len(commands)]
        if np.array_equal(applied, expected):
            raise RuntimeError("Deranged schedule contains an accidental match.")
    return schedule


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
