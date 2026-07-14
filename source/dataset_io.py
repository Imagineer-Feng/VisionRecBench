import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


DATASET_SCHEMA_VERSION = "1.0"


def utc_timestamp():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        while True:
            chunk = file_obj.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def image_descriptor(dataset_root, image_path):
    dataset_root = Path(dataset_root).resolve()
    image_path = Path(image_path).resolve()
    relative_path = image_path.relative_to(dataset_root).as_posix()
    with Image.open(image_path) as image:
        width, height = image.size
    return {
        "path": relative_path,
        "sha256": sha256_file(image_path),
        "width": int(width),
        "height": int(height),
    }


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")
    temporary.replace(path)


def atomic_write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, sort_keys=True))
            file_obj.write("\n")
    temporary.replace(path)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def load_manifest(dataset_root):
    dataset_root = Path(dataset_root)
    manifest_path = dataset_root / "manifest.jsonl"
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {manifest_path} at line {line_number}."
                ) from exc
    return rows


def episode_summary(record, dataset_root):
    record_path = (
        Path(dataset_root)
        / "episodes"
        / record["episode_id"]
        / "episode.json"
    )
    return {
        "schema_version": record["schema_version"],
        "episode_id": record["episode_id"],
        "episode_index": record["episode_index"],
        "episode_signature": record["episode_signature"],
        "scenario": record["scenario"],
        "scene": record["scene"],
        "seed": record["seed"],
        "target_present": record["target_present"],
        "target_index": record["target_index"],
        "answer_index": record["answer_index"],
        "record_path": record_path.relative_to(dataset_root).as_posix(),
        "record_sha256": sha256_file(record_path),
    }


def rebuild_manifest(dataset_root):
    dataset_root = Path(dataset_root)
    records = []
    for record_path in (dataset_root / "episodes").glob("*/episode.json"):
        record = load_json(record_path)
        records.append(episode_summary(record, dataset_root))
    records.sort(key=lambda item: (item["scenario"], item["episode_index"]))
    atomic_write_jsonl(dataset_root / "manifest.jsonl", records)
    return records


def dataset_content_hash(rows):
    payload = json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_episode(dataset_root, summary):
    return load_json(Path(dataset_root) / summary["record_path"])


def _validate_image(dataset_root, descriptor, verify_checksums, errors, prefix):
    dataset_root = Path(dataset_root).resolve()
    image_path = (dataset_root / descriptor.get("path", "")).resolve()
    try:
        image_path.relative_to(dataset_root)
    except ValueError:
        errors.append(f"{prefix}: image path escapes dataset root: {image_path}")
        return
    if not image_path.is_file():
        errors.append(f"{prefix}: missing image {image_path}")
        return
    try:
        with Image.open(image_path) as image:
            width, height = image.size
        if int(descriptor.get("width", -1)) != width:
            errors.append(f"{prefix}: width mismatch for {image_path}")
        if int(descriptor.get("height", -1)) != height:
            errors.append(f"{prefix}: height mismatch for {image_path}")
    except Exception as exc:
        errors.append(f"{prefix}: unreadable image {image_path}: {exc}")
        return
    if verify_checksums and sha256_file(image_path) != descriptor.get("sha256"):
        errors.append(f"{prefix}: checksum mismatch for {image_path}")


def validate_dataset(dataset_root, verify_checksums=True):
    dataset_root = Path(dataset_root)
    errors = []
    warnings = []
    metadata_path = dataset_root / "metadata.json"
    manifest_path = dataset_root / "manifest.jsonl"
    if not metadata_path.is_file():
        return {"valid": False, "errors": [f"Missing {metadata_path}"]}
    if not manifest_path.is_file():
        return {"valid": False, "errors": [f"Missing {manifest_path}"]}

    metadata = load_json(metadata_path)
    rows = load_manifest(dataset_root)
    if metadata.get("schema_version") != DATASET_SCHEMA_VERSION:
        errors.append("Unsupported dataset schema version.")

    identifiers = [row.get("episode_id") for row in rows]
    signatures = [row.get("episode_signature") for row in rows]
    duplicate_ids = [key for key, count in Counter(identifiers).items() if count > 1]
    duplicate_signatures = [
        key for key, count in Counter(signatures).items() if count > 1
    ]
    if duplicate_ids:
        errors.append(f"Duplicate episode IDs: {duplicate_ids[:5]}")
    if duplicate_signatures:
        errors.append(
            "Duplicate physical/visual episode signatures: "
            f"{duplicate_signatures[:5]}"
        )

    counts = Counter()
    answer_positions = defaultdict(Counter)
    target_positions = defaultdict(Counter)
    target_presence = defaultdict(Counter)
    mapping_options = defaultdict(Counter)
    behavior_positions = defaultdict(Counter)
    task_modes = {}
    for summary in rows:
        episode_id = summary.get("episode_id", "<unknown>")
        record_path = (
            dataset_root / summary.get("record_path", "")
        ).resolve()
        try:
            record_path.relative_to(dataset_root.resolve())
        except ValueError:
            errors.append(
                f"{episode_id}: record path escapes dataset root: {record_path}"
            )
            continue
        if not record_path.is_file():
            errors.append(f"{episode_id}: missing record {record_path}")
            continue
        try:
            record = load_json(record_path)
        except Exception as exc:
            errors.append(f"{episode_id}: unreadable record: {exc}")
            continue

        scenario = record.get("scenario")
        counts[scenario] += 1
        task_modes[scenario] = record.get("task", {}).get("task_mode")
        answer_positions[scenario][record.get("answer_index")] += 1
        target_positions[scenario][record.get("target_index")] += 1
        target_presence[scenario][record.get("target_present")] += 1

        visible_behavior = record.get("task", {}).get("visible_arm_behavior", {})
        sampled_mapping = visible_behavior.get("sampled_mapping_option")
        if sampled_mapping is not None:
            mapping_options[scenario][
                (record.get("target_present"), sampled_mapping)
            ] += 1
        for candidate in record.get("candidates", []):
            behavior_positions[scenario][
                (candidate.get("behavior"), candidate.get("index"))
            ] += 1

        if record.get("schema_version") != DATASET_SCHEMA_VERSION:
            errors.append(f"{episode_id}: unsupported record schema version")
        summary_fields = (
            "episode_id",
            "episode_index",
            "episode_signature",
            "scenario",
            "scene",
            "seed",
            "target_present",
            "target_index",
            "answer_index",
        )
        mismatched_fields = [
            key for key in summary_fields if summary.get(key) != record.get(key)
        ]
        if mismatched_fields:
            errors.append(
                f"{episode_id}: manifest/record mismatch for "
                + ", ".join(mismatched_fields)
            )
        if summary.get("record_sha256") != sha256_file(record_path):
            errors.append(f"{episode_id}: record checksum mismatch")
        expected_steps = int(record.get("task", {}).get("episode_steps", -1))
        if len(record.get("steps", [])) != expected_steps:
            errors.append(f"{episode_id}: unexpected number of steps")
        observed_step_numbers = [
            step.get("step") for step in record.get("steps", [])
        ]
        if observed_step_numbers != list(range(1, expected_steps + 1)):
            errors.append(f"{episode_id}: steps are not sequential")
        answer_options = record.get("answer_options", [])
        answer_index = record.get("answer_index")
        if (
            not isinstance(answer_index, int)
            or not 1 <= answer_index <= len(answer_options)
        ):
            errors.append(f"{episode_id}: invalid answer index/options")
        elif record.get("answer_text") != answer_options[answer_index - 1]:
            errors.append(f"{episode_id}: answer text does not match answer index")
        _validate_image(
            dataset_root,
            record.get("initial_observation", {}),
            verify_checksums,
            errors,
            episode_id,
        )
        for step in record.get("steps", []):
            step_prefix = f"{episode_id}/step-{step.get('step')}"
            _validate_image(
                dataset_root,
                step.get("observation", {}),
                verify_checksums,
                errors,
                step_prefix,
            )
            _validate_image(
                dataset_root,
                step.get("evidence", {}),
                verify_checksums,
                errors,
                step_prefix,
            )

    configured_count = int(metadata.get("episodes_per_scene", 0))
    for scenario in metadata.get("scenarios", []):
        if counts[scenario] != configured_count:
            errors.append(
                f"{scenario}: expected {configured_count} episodes, "
                f"found {counts[scenario]}"
            )
        values = list(answer_positions[scenario].values())
        if values and max(values) - min(values) > 1:
            errors.append(f"{scenario}: answer positions are not balanced")
        if task_modes.get(scenario) == "single_binary":
            values = list(target_presence[scenario].values())
            if len(values) != 2 or max(values) - min(values) > 1:
                errors.append(f"{scenario}: binary conditions are not balanced")
        elif task_modes.get(scenario) == "multi_arm":
            values = list(target_positions[scenario].values())
            if values and max(values) - min(values) > 1:
                errors.append(f"{scenario}: target positions are not balanced")

        values = list(mapping_options[scenario].values())
        if values and max(values) - min(values) > 1:
            errors.append(f"{scenario}: mapping conditions are not balanced")
        values = list(behavior_positions[scenario].values())
        if task_modes.get(scenario) == "multi_arm" and values:
            if max(values) - min(values) > 1:
                errors.append(f"{scenario}: behavior positions are not balanced")

    current_hash = dataset_content_hash(rows)
    stored_hash = metadata.get("content_sha256")
    if not stored_hash:
        errors.append("metadata is missing content_sha256")
    elif stored_hash != current_hash:
        errors.append("metadata content_sha256 does not match the manifest")

    return {
        "valid": not errors,
        "schema_version": metadata.get("schema_version"),
        "dataset_name": metadata.get("dataset_name"),
        "content_sha256": current_hash,
        "episode_count": len(rows),
        "scenario_counts": dict(counts),
        "answer_positions": {
            scenario: {str(key): value for key, value in counter.items()}
            for scenario, counter in answer_positions.items()
        },
        "target_positions": {
            scenario: {str(key): value for key, value in counter.items()}
            for scenario, counter in target_positions.items()
        },
        "target_presence": {
            scenario: {str(key): value for key, value in counter.items()}
            for scenario, counter in target_presence.items()
        },
        "mapping_options": {
            scenario: {str(key): value for key, value in counter.items()}
            for scenario, counter in mapping_options.items()
        },
        "behavior_positions": {
            scenario: {str(key): value for key, value in counter.items()}
            for scenario, counter in behavior_positions.items()
        },
        "errors": errors,
        "warnings": warnings,
    }
