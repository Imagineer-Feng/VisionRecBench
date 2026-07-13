import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import numpy as np
from PIL import Image


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from scripts.inference import (  # noqa: E402
    build_model_content,
    build_prompts,
    get_control_labels,
    validate_api_key,
)
from source.agent import create_identifier  # noqa: E402
from source.dataset_io import (  # noqa: E402
    atomic_write_json,
    load_episode,
    load_json,
    load_manifest,
    utc_timestamp,
    validate_dataset,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen VisionRecBench episodes without starting Isaac Sim."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--level",
        type=int,
        nargs="+",
        choices=[0, 1, 2, 3],
        default=[1],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "results" / "offline",
    )
    parser.add_argument("--scenario", nargs="+", default=None)
    parser.add_argument(
        "--max-image-history",
        type=int,
        default=-1,
        help="Previous evidence images to include; -1 uses the complete episode.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-checksums", action="store_true")
    parser.add_argument("--random-seed", type=int, default=0)
    args = parser.parse_args()
    if args.max_image_history < -1:
        parser.error("--max-image-history must be -1 or non-negative")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


def load_rgb(dataset_root, descriptor):
    image_path = Path(dataset_root) / descriptor["path"]
    with Image.open(image_path) as image:
        return np.asarray(image.convert("RGB")).copy()


def content_sha256(content_items):
    digest = hashlib.sha256()
    for item in content_items:
        if isinstance(item, str):
            digest.update(b"text\0")
            digest.update(item.encode("utf-8"))
        elif isinstance(item, np.ndarray):
            digest.update(b"image\0")
            digest.update(str(item.shape).encode("ascii"))
            digest.update(str(item.dtype).encode("ascii"))
            digest.update(item.tobytes())
        else:
            raise TypeError(f"Unsupported content item: {type(item)!r}")
    return digest.hexdigest()


def sanitized_api_endpoint(model):
    if model == "random":
        return None
    raw_url = (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("BASE_URL")
        or "https://api.openai.com/v1"
    )
    parsed = urlsplit(raw_url)
    hostname = parsed.hostname or ""
    netloc = hostname
    if parsed.port is not None:
        netloc = f"{hostname}:{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def build_episode_content(record, level, max_image_history, dataset_root):
    task = record["task"]
    steps = record["steps"]
    if not steps:
        raise ValueError(f"Episode {record['episode_id']} has no steps.")

    prompt_prefix, prompt_suffix = build_prompts(
        level,
        task,
        max_image_history=(
            len(steps) if max_image_history < 0 else max_image_history
        ),
        answer_options=record["answer_options"],
    )
    command_history = [step["command"] for step in steps]
    prior_steps = steps[:-1]
    if max_image_history >= 0:
        prior_steps = prior_steps[-max_image_history:] if max_image_history else []

    visual_history = [
        load_rgb(dataset_root, step["evidence"])
        for step in prior_steps
    ]
    visual_history_commands = [step["command"] for step in prior_steps]
    final_step = steps[-1]
    current_evidence = load_rgb(dataset_root, final_step["evidence"])
    current_observation = load_rgb(dataset_root, final_step["observation"])
    control_labels = record.get("control_labels") or get_control_labels(task)

    return build_model_content(
        prompt_prefix,
        prompt_suffix,
        control_labels,
        final_step["command"],
        command_history,
        visual_history,
        current_evidence,
        current_observation,
        judgement_interval=int(task.get("judge_interval_steps", 1)),
        visual_history_mode=record["visual_history_mode"],
        visual_history_commands=visual_history_commands,
    )


def result_path(output_root, metadata, model, level, record):
    model_dir = model.replace("/", "-")
    dataset_dir = metadata["dataset_name"].replace("/", "-")
    return (
        Path(output_root)
        / dataset_dir
        / f"prompt_level{level}"
        / model_dir
        / record["scenario"]
        / f"{record['episode_id']}.json"
    )


def evaluate_one(
    identifier,
    record,
    content_items,
    metadata,
    model,
    level,
    random_seed=None,
):
    identification = identifier.identify(
        content_items,
        len(record["answer_options"]),
    )
    correct = identification.choice == int(record["answer_index"])
    return {
        "schema_version": "1.0",
        "dataset_name": metadata["dataset_name"],
        "dataset_content_sha256": metadata["content_sha256"],
        "episode_id": record["episode_id"],
        "episode_signature": record["episode_signature"],
        "scenario": record["scenario"],
        "scene": record["scene"],
        "seed": record["seed"],
        "model": model,
        "api_endpoint": sanitized_api_endpoint(model),
        "prompt_level": level,
        "temperature": 0.0 if model != "random" else None,
        "random_seed": random_seed if model == "random" else None,
        "evaluated_at": utc_timestamp(),
        "input_content_sha256": content_sha256(content_items),
        "target_present": record["target_present"],
        "target_index": record["target_index"],
        "answer_index": record["answer_index"],
        "answer_text": record["answer_text"],
        "answer_options": record["answer_options"],
        "choice": identification.choice,
        "valid": identification.valid,
        "correct": correct,
        "response_text": identification.text,
        "bad_response": 0 if identification.valid else 1,
    }


def main():
    args = parse_args()
    dataset_root = args.dataset.resolve()
    report = validate_dataset(
        dataset_root,
        verify_checksums=args.verify_checksums,
    )
    if not report["valid"]:
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(1)

    metadata = load_json(dataset_root / "metadata.json")
    rows = load_manifest(dataset_root)
    if args.scenario:
        selected = set(args.scenario)
        rows = [row for row in rows if row["scenario"] in selected]
    if args.limit is not None:
        rows = rows[: args.limit]

    validate_api_key(args.model)
    random.seed(args.random_seed)
    identifier = create_identifier(args.model)
    outcomes = Counter()
    total = len(rows) * len(args.level)
    current = 0
    for level in args.level:
        for summary in rows:
            current += 1
            record = load_episode(dataset_root, summary)
            content_items = build_episode_content(
                record,
                level,
                args.max_image_history,
                dataset_root,
            )
            expected_input_hash = content_sha256(content_items)
            output_path = result_path(
                args.output,
                metadata,
                args.model,
                level,
                record,
            )
            if output_path.is_file():
                if not args.resume:
                    raise FileExistsError(
                        f"Result already exists: {output_path}. Use --resume."
                    )
                existing = load_json(output_path)
                expected_existing = {
                    "dataset_content_sha256": metadata["content_sha256"],
                    "episode_id": record["episode_id"],
                    "episode_signature": record["episode_signature"],
                    "model": args.model,
                    "prompt_level": level,
                    "input_content_sha256": expected_input_hash,
                    "random_seed": (
                        args.random_seed if args.model == "random" else None
                    ),
                }
                mismatches = [
                    key
                    for key, value in expected_existing.items()
                    if existing.get(key) != value
                ]
                if mismatches:
                    raise ValueError(
                        f"Existing result metadata mismatch ({', '.join(mismatches)}): "
                        f"{output_path}"
                    )
                print(
                    f"[{current}/{total}] skipping {record['episode_id']} L{level}",
                    flush=True,
                )
                outcomes["skipped"] += 1
                continue

            print(
                f"[{current}/{total}] evaluating {record['episode_id']} L{level}",
                flush=True,
            )
            if args.model == "random":
                per_episode_seed = int.from_bytes(
                    hashlib.sha256(
                        f"{args.random_seed}:{level}:{record['episode_id']}".encode(
                            "utf-8"
                        )
                    ).digest()[:8],
                    byteorder="big",
                )
                random.seed(per_episode_seed)
            result = evaluate_one(
                identifier,
                record,
                content_items,
                metadata,
                args.model,
                level,
                random_seed=(
                    args.random_seed if args.model == "random" else None
                ),
            )
            atomic_write_json(output_path, result)
            outcomes["completed"] += 1
            outcomes["correct" if result["correct"] else "incorrect"] += 1
            if not result["valid"]:
                outcomes["invalid"] += 1

    print(json.dumps(dict(outcomes), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
