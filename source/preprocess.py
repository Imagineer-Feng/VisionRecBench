import copy
import json
from pathlib import Path


TASK_DIR = Path(__file__).resolve().parents[1] / "tasks"


def _load_repo(name):
    with open(TASK_DIR / f"{name}_repo.json", "r") as f:
        return json.load(f)


repos = {
    "arm": _load_repo("arm"),
    "distractor": _load_repo("distractor"),
    "scenario": _load_repo("scenario"),
}


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


def construct(id_dict):
    scenario_name = id_dict["scenario"]
    task_dict = copy.deepcopy(repos["scenario"][scenario_name])
    task_dict["name"] = scenario_name

    if "arm" in id_dict and id_dict["arm"] is not None:
        task_dict["arm"] = id_dict["arm"]

    task_dict = preprocess(task_dict)

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
