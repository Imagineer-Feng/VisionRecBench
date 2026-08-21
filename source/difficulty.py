import copy


DIFFICULTY_LEVELS = (1, 2, 3)
DIFFICULTY_NAMES = {
    1: "easy",
    2: "medium",
    3: "hard",
}


def _resolve_level_values(value, level):
    if isinstance(value, list):
        return [_resolve_level_values(item, level) for item in value]
    if not isinstance(value, dict):
        return copy.deepcopy(value)

    resolved = {}
    deferred = []
    for key, item in value.items():
        if key.endswith("_by_level"):
            deferred.append((key, item))
        else:
            resolved[key] = _resolve_level_values(item, level)

    for key, options in deferred:
        if not isinstance(options, dict) or str(level) not in options:
            raise ValueError(
                f"{key} must define a value for difficulty level {level}."
            )
        base_key = key[: -len("_by_level")]
        resolved[base_key] = _resolve_level_values(options[str(level)], level)
    return resolved


def apply_difficulty_level(task_dict, level):
    level = int(level)
    if level not in DIFFICULTY_LEVELS:
        raise ValueError(
            f"difficulty level must be one of {DIFFICULTY_LEVELS}, got {level}"
        )
    resolved = _resolve_level_values(task_dict, level)
    resolved["difficulty_level"] = level
    resolved["difficulty_name"] = DIFFICULTY_NAMES[level]
    return resolved
