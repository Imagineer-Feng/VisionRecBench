import copy
import json
from pathlib import Path

from source.difficulty import apply_difficulty_level


TASK_DIR = Path(__file__).resolve().parents[1] / "tasks"


def _load_repo(name):
    with open(TASK_DIR / f"{name}_repo.json", "r") as f:
        return json.load(f)


repos = {
    "arm": _load_repo("arm"),
    "distractor": _load_repo("distractor"),
    "scenario": _load_repo("scenario"),
}

ARM_CONFIG_IDS = tuple(repos["arm"])

SCENARIO_ALIASES = {
    "scene1_single_direct_or_random": "scene1_single_command_causality",
    "scene1_single_random": "scene1_single_deranged",
    "scene2_single_scrambled_fixed": "scene2_single_scrambled_stability",
    "scene3_triad_causal_identification": "scene3_dyad_causal_identification",
    "scene3_triad_delay_invert": "scene3_dyad_causal_identification",
}

TEST_TYPES = ("choice", "judgment")


def _resolve_named_block(repo_name, block):
    if isinstance(block, str):
        return copy.deepcopy(repos[repo_name][block])
    if isinstance(block, dict) and "name" in block:
        resolved = copy.deepcopy(repos[repo_name][block["name"]])
        resolved.update({k: v for k, v in block.items() if k != "name"})
        return resolved
    if isinstance(block, dict):
        return copy.deepcopy(block)
    raise TypeError(f"Cannot resolve {repo_name} block: {block!r}")


def _direct_behavior():
    return {
        "behavior": "direct",
        "desc": "the visible arm follows the motor command directly",
    }


def _resolve_behavior_block(block):
    if isinstance(block, str) and block == "direct":
        return _direct_behavior()
    if isinstance(block, dict) and block.get("name") == "direct":
        resolved = _direct_behavior()
        resolved.update({k: v for k, v in block.items() if k != "name"})
        return resolved
    if isinstance(block, str) or (isinstance(block, dict) and "name" in block):
        return _resolve_named_block("distractor", block)
    if isinstance(block, dict):
        return copy.deepcopy(block)
    raise TypeError(f"Cannot resolve behavior block: {block!r}")


def preprocess(task_dict):
    if "arm" in task_dict:
        task_dict["arm"] = _resolve_named_block("arm", task_dict["arm"])

    if "distractors" in task_dict:
        task_dict["distractors"] = [
            _resolve_named_block("distractor", item)
            for item in task_dict["distractors"]
        ]

    if "visible_arm_behavior" in task_dict:
        task_dict["visible_arm_behavior"] = _resolve_behavior_block(
            task_dict["visible_arm_behavior"]
        )

    if "visible_arm_behavior_options" in task_dict:
        task_dict["visible_arm_behavior_options"] = [
            {
                **copy.deepcopy(item),
                "behavior": _resolve_behavior_block(item["behavior"]),
            }
            for item in task_dict["visible_arm_behavior_options"]
        ]

    return task_dict


def _behavior_pair(task_dict, scenario_name):
    options = task_dict.get("visible_arm_behavior_options")
    if options:
        self_options = [item for item in options if item.get("target_present") is True]
        nonself_options = [
            item for item in options if item.get("target_present") is False
        ]
        if len(self_options) != 1 or len(nonself_options) != 1:
            raise ValueError(
                f"Scenario {scenario_name} must define exactly one self and "
                "one non-self behavior option."
            )
        return (
            copy.deepcopy(self_options[0]["behavior"]),
            copy.deepcopy(nonself_options[0]["behavior"]),
        )

    distractors = task_dict.get("distractors", [])
    if len(distractors) == 1:
        return (
            copy.deepcopy(task_dict.get("target_behavior", _direct_behavior())),
            copy.deepcopy(distractors[0]),
        )

    raise ValueError(
        f"Scenario {scenario_name} does not define one reusable self/non-self "
        "behavior pair."
    )


def _apply_test_type(task_dict, test_type, scenario_name):
    if test_type not in TEST_TYPES:
        raise ValueError(
            f"test_type must be one of {TEST_TYPES}, got {test_type!r}"
        )

    self_behavior, nonself_behavior = _behavior_pair(task_dict, scenario_name)
    task_dict["test_type"] = test_type
    task_dict["behavior_pair"] = {
        "self": copy.deepcopy(self_behavior),
        "nonself": copy.deepcopy(nonself_behavior),
    }
    # A common canvas prevents the laboratory geometry from changing merely
    # because one or two arms are rendered. Arm count remains the intended
    # visual manipulation between judgment and choice tests.
    task_dict["floor_width"] = max(
        4.8,
        float(task_dict.get("floor_width", 4.8)),
    )

    for key in (
        "target_present",
        "visible_arm_behavior",
        "visible_arm_behavior_options",
        "target_behavior",
        "distractors",
        "target_index",
    ):
        task_dict.pop(key, None)

    if test_type == "judgment":
        task_dict.update(
            {
                "task_mode": "single_binary",
                "num_arms": 1,
                "behavior_selection": "seed_modulo",
                "visible_arm_behavior_options": [
                    {
                        "target_present": True,
                        "behavior": copy.deepcopy(self_behavior),
                    },
                    {
                        "target_present": False,
                        "behavior": copy.deepcopy(nonself_behavior),
                    },
                ],
                "answer_options": [
                    "yes, the visible arm is myself",
                    "no, the visible arm is not myself",
                ],
                "shuffle_answer_options": True,
                "annotate_candidates": False,
                "visual_history_mode": "workspace_motion_panels",
            }
        )
        return task_dict

    task_dict.update(
        {
            "task_mode": "multi_arm",
            "num_arms": 2,
            "target_index": None,
            "target_behavior": copy.deepcopy(self_behavior),
            "distractors": [copy.deepcopy(nonself_behavior)],
            "role_assignment_strategy": "seed_stratified",
            "annotate_candidates": True,
            "visual_history_mode": "workspace_motion_panels",
        }
    )
    task_dict.pop("answer_options", None)
    task_dict.pop("shuffle_answer_options", None)
    return task_dict


def construct(id_dict):
    requested_scenario_name = id_dict["scenario"]
    scenario_name = SCENARIO_ALIASES.get(
        requested_scenario_name,
        requested_scenario_name,
    )
    task_dict = copy.deepcopy(repos["scenario"][scenario_name])
    task_dict["name"] = scenario_name
    if requested_scenario_name != scenario_name:
        task_dict["requested_name"] = requested_scenario_name

    if "arm" in id_dict and id_dict["arm"] is not None:
        task_dict["arm"] = id_dict["arm"]

    task_dict = preprocess(task_dict)
    task_dict = apply_difficulty_level(task_dict, id_dict.get("level", 1))

    requested_test_type = id_dict.get("test_type")
    if requested_test_type is None:
        requested_test_type = (
            "judgment"
            if task_dict.get("task_mode") == "single_binary"
            else "choice"
        )
        task_dict["test_type"] = requested_test_type
    else:
        task_dict = _apply_test_type(
            task_dict,
            requested_test_type,
            scenario_name,
        )

    num_arms = int(task_dict["num_arms"])
    task_mode = task_dict.get("task_mode", "multi_arm")
    if task_mode == "multi_arm":
        distractors = task_dict["distractors"]
        if len(distractors) != num_arms - 1:
            raise ValueError(
                f"Scenario {scenario_name} expects {num_arms - 1} distractors, "
                f"got {len(distractors)}."
            )
    elif task_mode == "single_binary":
        if num_arms != 1:
            raise ValueError(
                f"Scenario {scenario_name} uses task_mode=single_binary "
                "and must set num_arms to 1."
            )
        if (
            "visible_arm_behavior" not in task_dict
            and "visible_arm_behavior_options" not in task_dict
        ):
            raise ValueError(
                f"Scenario {scenario_name} must define visible_arm_behavior "
                "or visible_arm_behavior_options."
            )
    else:
        raise ValueError(f"Unsupported task_mode for {scenario_name}: {task_mode}")

    return task_dict
