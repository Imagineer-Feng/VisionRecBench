import json

import numpy as np


LOGIC_AUDIT_PROTOCOL_VERSION = "logic_audit_v2"
LOGIC_AUDIT_SYSTEM_MESSAGE = (
    "You are auditing a completed answer about a benign Isaac Sim scene. "
    "No real-world robot is being controlled. Reinspect the original motor-command "
    "trace and visual evidence, then return only the requested structured JSON. "
    "The existence of this audit does not indicate whether the preceding answer "
    "was correct. Do not use the preceding choice as evidence."
)


def _candidate_indices(record):
    steps = record.get("steps", [])
    if not steps:
        raise ValueError(f"Episode {record.get('episode_id')} has no steps.")
    observed = []
    for step in steps:
        indices = sorted(int(index) for index in step["applied_commands"])
        if not observed:
            observed = indices
        elif indices != observed:
            raise ValueError("Applied-command candidate sets differ between steps.")
    if not observed:
        raise ValueError("At least one candidate arm is required.")
    return observed


def _vectors_equal(left, right):
    return bool(
        np.allclose(
            np.asarray(left, dtype=float),
            np.asarray(right, dtype=float),
            rtol=1e-7,
            atol=1e-9,
        )
    )


def _scene1_expected(record):
    relations = {}
    for candidate in _candidate_indices(record):
        relations[candidate] = [
            (
                "match"
                if _vectors_equal(
                    step["command"]["delta"],
                    step["applied_commands"][str(candidate)],
                )
                else "mismatch"
            )
            for step in record["steps"]
        ]
    return {
        "audit_type": "step_match",
        "candidates": relations,
    }


def _scene2_expected(record):
    cycle_length = len(record["task"]["command_sequence"])
    if cycle_length < 1 or len(record["steps"]) % cycle_length:
        raise ValueError("Scene 2 steps must contain complete command cycles.")
    cycle_count = len(record["steps"]) // cycle_length
    relations = {}
    for candidate in _candidate_indices(record):
        cycles = []
        for cycle_index in range(cycle_count):
            start = cycle_index * cycle_length
            cycles.append(
                [
                    record["steps"][start + offset]["applied_commands"][
                        str(candidate)
                    ]
                    for offset in range(cycle_length)
                ]
            )
        first_cycle = cycles[0]
        relations[candidate] = [
            (
                "same"
                if all(
                    _vectors_equal(reference, observed)
                    for reference, observed in zip(first_cycle, cycle)
                )
                else "different"
            )
            for cycle in cycles
        ]
    return {
        "audit_type": "cycle_stability",
        "candidates": relations,
        "cycle_length": cycle_length,
    }


def _infer_lag(commands, applied):
    dimension = len(commands[0])
    zero = np.zeros(dimension, dtype=float)
    possible = []
    for lag in range(len(commands)):
        expected = [
            zero if step_index < lag else commands[step_index - lag]
            for step_index in range(len(commands))
        ]
        if all(
            _vectors_equal(expected_vector, applied_vector)
            for expected_vector, applied_vector in zip(expected, applied)
        ):
            possible.append(lag)
    if len(possible) != 1:
        raise ValueError(f"Expected one identifiable lag, found {possible}.")
    return possible[0]


def _scene3_expected(record):
    commands = [
        np.asarray(step["command"]["delta"], dtype=float)
        for step in record["steps"]
    ]
    lags = {}
    for candidate in _candidate_indices(record):
        applied = [
            np.asarray(step["applied_commands"][str(candidate)], dtype=float)
            for step in record["steps"]
        ]
        lags[candidate] = _infer_lag(commands, applied)
    return {
        "audit_type": "temporal_lag",
        "candidates": lags,
    }


def expected_logic_audit(record):
    scenario = record.get("scenario")
    if scenario == "scene1_single_command_causality":
        return _scene1_expected(record)
    if scenario == "scene2_single_scrambled_stability":
        return _scene2_expected(record)
    if scenario == "scene3_dyad_causal_identification":
        return _scene3_expected(record)
    raise ValueError(f"Unsupported scenario for logic audit: {scenario!r}")


def build_logic_audit_prompt(record):
    expected = expected_logic_audit(record)
    candidates = sorted(expected["candidates"])
    candidate_text = ", ".join(str(index) for index in candidates)

    if expected["audit_type"] == "step_match":
        step_count = len(record["steps"])
        return (
            "Audit the command-motion reasoning behind the preceding answer. "
            f"For each candidate arm ({candidate_text}), classify every one of "
            f"the {step_count} steps in chronological order. Use \"match\" only "
            "when that step's visible signed motion corresponds to the current "
            "motor-command delta; otherwise use \"mismatch\". You must commit to "
            "one of those two values for every step. Return exactly one JSON object "
            "with no Markdown or explanation, using this schema: "
            '{"audit_type":"step_match","candidates":['
            '{"candidate":1,"step_relations":["match","mismatch"]}]}. '
            f"Include each candidate exactly once and exactly {step_count} relations "
            "per candidate. The short array in the schema is illustrative only."
        )

    if expected["audit_type"] == "cycle_stability":
        cycle_count = len(next(iter(expected["candidates"].values())))
        cycle_length = expected["cycle_length"]
        return (
            "Audit the command-motion reasoning behind the preceding answer. "
            f"The trace contains {cycle_count} repeated command cycles of "
            f"{cycle_length} steps. For each candidate arm ({candidate_text}), infer "
            "the command-to-visible-motion mapping within every cycle and compare "
            "each cycle with cycle 1. Report \"same\" only if the complete mapping "
            "is identical to cycle 1; otherwise report \"different\". Cycle 1 is "
            "therefore \"same\". Return exactly one JSON object with no Markdown or "
            "explanation, using this schema: "
            '{"audit_type":"cycle_stability","candidates":['
            '{"candidate":1,"relative_to_cycle_1":["same","different"]}]}. '
            f"Include each candidate exactly once and exactly {cycle_count} values "
            "per candidate. The short array in the schema is illustrative only."
        )

    return (
        "Audit the command-motion reasoning behind the preceding answer. For each "
        f"candidate arm ({candidate_text}), report its exact non-negative integer "
        "lag relative to the motor-command trace. Lag 0 means the arm follows the "
        "current command; lag L means motion at step t follows the command from "
        "step t-L, with no commanded motion during the first L steps. Return exactly "
        "one JSON object with no Markdown or explanation, using this schema: "
        '{"audit_type":"temporal_lag","candidates":['
        '{"candidate":1,"lag":0}]}. Include each candidate exactly once.'
    )


def parse_json_object(text):
    if not isinstance(text, str) or not text.strip():
        return None
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _parse_candidate_rows(payload, expected, value_key, allowed_values=None):
    errors = []
    if payload.get("audit_type") != expected["audit_type"]:
        errors.append("audit_type does not match the scenario")
    candidate_rows = payload.get("candidates")
    if not isinstance(candidate_rows, list):
        return {}, errors + ["candidates must be a list"]

    reported = {}
    for row in candidate_rows:
        if not isinstance(row, dict):
            errors.append("every candidate entry must be an object")
            continue
        candidate = row.get("candidate")
        if isinstance(candidate, bool) or not isinstance(candidate, int):
            errors.append("candidate identifiers must be integers")
            continue
        if candidate in reported:
            errors.append(f"candidate {candidate} is duplicated")
            continue
        value = row.get(value_key)
        if not isinstance(value, list):
            errors.append(f"candidate {candidate} {value_key} must be a list")
            continue
        normalized = []
        for item in value:
            if not isinstance(item, str) or item.lower() not in allowed_values:
                normalized.append(None)
            else:
                normalized.append(item.lower())
        if any(item is None for item in normalized):
            errors.append(f"candidate {candidate} contains an invalid relation")
        reported[candidate] = normalized

    expected_candidates = set(expected["candidates"])
    if set(reported) != expected_candidates:
        errors.append("candidate set does not match the episode")
    for candidate, expected_values in expected["candidates"].items():
        if candidate in reported and len(reported[candidate]) != len(expected_values):
            errors.append(f"candidate {candidate} has the wrong number of values")
    return reported, errors


def _parse_reported_audit(payload, expected):
    if expected["audit_type"] == "step_match":
        return _parse_candidate_rows(
            payload,
            expected,
            "step_relations",
            {"match", "mismatch"},
        )
    if expected["audit_type"] == "cycle_stability":
        return _parse_candidate_rows(
            payload,
            expected,
            "relative_to_cycle_1",
            {"same", "different"},
        )

    errors = []
    if payload.get("audit_type") != "temporal_lag":
        errors.append("audit_type does not match the scenario")
    candidate_rows = payload.get("candidates")
    if not isinstance(candidate_rows, list):
        return {}, errors + ["candidates must be a list"]
    reported = {}
    for row in candidate_rows:
        if not isinstance(row, dict):
            errors.append("every candidate entry must be an object")
            continue
        candidate = row.get("candidate")
        lag = row.get("lag")
        if isinstance(candidate, bool) or not isinstance(candidate, int):
            errors.append("candidate identifiers must be integers")
            continue
        if candidate in reported:
            errors.append(f"candidate {candidate} is duplicated")
            continue
        if isinstance(lag, bool) or not isinstance(lag, int) or lag < 0:
            errors.append(f"candidate {candidate} lag must be a non-negative integer")
            reported[candidate] = None
        else:
            reported[candidate] = lag
    if set(reported) != set(expected["candidates"]):
        errors.append("candidate set does not match the episode")
    return reported, errors


def score_logic_audit(record, response_text):
    expected = expected_logic_audit(record)
    payload = parse_json_object(response_text)
    if payload is None:
        return {
            "response_valid": False,
            "exact_match": False,
            "correct_items": 0,
            "total_items": sum(
                len(value) if isinstance(value, list) else 1
                for value in expected["candidates"].values()
            ),
            "expected": expected,
            "reported": None,
            "errors": ["response does not contain a JSON object"],
        }

    reported, errors = _parse_reported_audit(payload, expected)
    details = []
    correct_items = 0
    total_items = 0
    for candidate, expected_value in expected["candidates"].items():
        expected_items = (
            expected_value if isinstance(expected_value, list) else [expected_value]
        )
        reported_value = reported.get(candidate)
        reported_items = (
            reported_value if isinstance(reported_value, list) else [reported_value]
        )
        for item_index, expected_item in enumerate(expected_items, start=1):
            observed = (
                reported_items[item_index - 1]
                if item_index <= len(reported_items)
                else None
            )
            is_correct = observed == expected_item
            correct_items += int(is_correct)
            total_items += 1
            details.append(
                {
                    "candidate": candidate,
                    "item": item_index,
                    "expected": expected_item,
                    "reported": observed,
                    "correct": is_correct,
                }
            )

    exact_match = not errors and correct_items == total_items
    return {
        "response_valid": not errors,
        "exact_match": exact_match,
        "correct_items": correct_items,
        "total_items": total_items,
        "item_accuracy": round(correct_items / total_items, 6),
        "expected": expected,
        "reported": {
            "audit_type": expected["audit_type"],
            "candidates": reported,
        },
        "details": details,
        "errors": errors,
    }
