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


def preprocess(task_dict):
    if "arm" in task_dict:
        task_dict["arm"] = _resolve_named_block("arm", task_dict["arm"])

    if "distractors" in task_dict:
        task_dict["distractors"] = [
            _resolve_named_block("distractor", item)
            for item in task_dict["distractors"]
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
    distractors = task_dict["distractors"]
    if len(distractors) != num_arms - 1:
        raise ValueError(
            f"Scenario {scenario_name} expects {num_arms - 1} distractors, "
            f"got {len(distractors)}."
        )

    return task_dict
