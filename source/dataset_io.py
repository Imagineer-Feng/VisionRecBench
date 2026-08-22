import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


DATASET_SCHEMA_VERSION = "3.0"
SUPPORTED_DATASET_SCHEMA_VERSIONS = {"1.0", "2.0", DATASET_SCHEMA_VERSION}


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
    summary = {
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
    if "difficulty_level" in record:
        summary["difficulty_level"] = int(record["difficulty_level"])
        summary["difficulty_name"] = record["difficulty_name"]
    if "test_type" in record:
        summary["test_type"] = record["test_type"]
    if "nuisance_pair_id" in record:
        summary["nuisance_pair_id"] = record["nuisance_pair_id"]
        summary["nuisance_signature"] = record["nuisance_signature"]
    if "environment_template" in record:
        summary["environment_template"] = record["environment_template"]
    return summary


def rebuild_manifest(dataset_root):
    dataset_root = Path(dataset_root)
    records = []
    for record_path in (dataset_root / "episodes").glob("*/episode.json"):
        record = load_json(record_path)
        records.append(episode_summary(record, dataset_root))
    records.sort(
        key=lambda item: (
            item["scenario"],
            int(item.get("difficulty_level", 0)),
            item.get("test_type", ""),
            item["episode_index"],
        )
    )
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
    if metadata.get("schema_version") not in SUPPORTED_DATASET_SCHEMA_VERSIONS:
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
    nuisance_pair_signatures = defaultdict(set)
    nuisance_pair_conditions = defaultdict(Counter)
    nuisance_pair_ids_by_group = defaultdict(set)
    environment_template_counts = defaultdict(Counter)
    nuisance_pair_templates = defaultdict(set)
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
        difficulty_level = record.get("difficulty_level")
        test_type = record.get("test_type")
        if metadata.get("schema_version") == DATASET_SCHEMA_VERSION:
            if difficulty_level not in metadata.get("difficulty_levels", []):
                errors.append(f"{episode_id}: invalid difficulty level")
            if not record.get("difficulty_name"):
                errors.append(f"{episode_id}: missing difficulty name")
            if test_type not in metadata.get("test_types", []):
                errors.append(f"{episode_id}: invalid test type")
            if record.get("task", {}).get("test_type") != test_type:
                errors.append(f"{episode_id}: task/record test type mismatch")
            task_mode = record.get("task", {}).get("task_mode")
            num_arms = record.get("task", {}).get("num_arms")
            expected_mode = (
                "multi_arm" if test_type == "choice" else "single_binary"
            )
            expected_arms = 2 if test_type == "choice" else 1
            if task_mode != expected_mode or num_arms != expected_arms:
                errors.append(
                    f"{episode_id}: test type does not match task mode/arm count"
                )
        paired_nuisance = bool(metadata.get("paired_nuisance", False))
        nuisance_pair_id = record.get("nuisance_pair_id")
        nuisance_signature = record.get("nuisance_signature")
        environment_template = record.get("environment_template")
        if paired_nuisance:
            if not nuisance_pair_id:
                errors.append(f"{episode_id}: missing nuisance pair ID")
            if not nuisance_signature:
                errors.append(f"{episode_id}: missing nuisance signature")
            task = record.get("task", {})
            if task.get("nuisance_pair_id") != nuisance_pair_id:
                errors.append(f"{episode_id}: task/record nuisance pair mismatch")
            if task.get("nuisance_signature") != nuisance_signature:
                errors.append(f"{episode_id}: task/record nuisance signature mismatch")
        configured_templates = metadata.get("environment_templates", [])
        if configured_templates:
            if environment_template not in configured_templates:
                errors.append(f"{episode_id}: invalid environment template")
            if (
                record.get("task", {}).get("environment", {}).get("id")
                != environment_template
            ):
                errors.append(
                    f"{episode_id}: task/record environment template mismatch"
                )
        group_parts = [str(scenario)]
        if difficulty_level is not None:
            group_parts.append(f"level{difficulty_level}")
        if test_type is not None:
            group_parts.append(str(test_type))
        group = "/".join(group_parts)
        counts[group] += 1
        task_modes[group] = record.get("task", {}).get("task_mode")
        answer_positions[group][record.get("answer_index")] += 1
        target_positions[group][record.get("target_index")] += 1
        target_presence[group][record.get("target_present")] += 1

        visible_behavior = record.get("task", {}).get("visible_arm_behavior", {})
        sampled_mapping = visible_behavior.get("sampled_mapping_option")
        if sampled_mapping is not None:
            mapping_options[group][
                (record.get("target_present"), sampled_mapping)
            ] += 1
        for candidate in record.get("candidates", []):
            behavior_positions[group][
                (candidate.get("behavior"), candidate.get("index"))
            ] += 1
            sampled_mapping = candidate.get("sampled_mapping_option")
            if sampled_mapping is not None:
                mapping_options[group][
                    (candidate.get("role"), sampled_mapping)
                ] += 1
        if environment_template is not None:
            environment_template_counts[group][environment_template] += 1

        if record.get("schema_version") not in SUPPORTED_DATASET_SCHEMA_VERSIONS:
            errors.append(f"{episode_id}: unsupported record schema version")
        summary_fields = [
            "episode_id",
            "episode_index",
            "episode_signature",
            "scenario",
            "scene",
            "seed",
            "target_present",
            "target_index",
            "answer_index",
        ]
        if difficulty_level is not None:
            summary_fields.extend(("difficulty_level", "difficulty_name"))
        if test_type is not None:
            summary_fields.append("test_type")
        if nuisance_pair_id is not None:
            summary_fields.extend(("nuisance_pair_id", "nuisance_signature"))
        if environment_template is not None:
            summary_fields.append("environment_template")
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

        if paired_nuisance and nuisance_pair_id and nuisance_signature:
            cross_level_key = (scenario, nuisance_pair_id)
            nuisance_pair_signatures[cross_level_key].add(nuisance_signature)
            within_level_key = (
                scenario,
                difficulty_level,
                test_type,
                nuisance_pair_id,
            )
            nuisance_pair_ids_by_group[
                (scenario, difficulty_level, test_type)
            ].add(
                nuisance_pair_id
            )
            if environment_template is not None:
                nuisance_pair_templates[cross_level_key].add(
                    environment_template
                )
            if record.get("task", {}).get("task_mode") == "single_binary":
                condition = bool(record.get("target_present"))
            else:
                condition = record.get("target_index")
            nuisance_pair_conditions[within_level_key][condition] += 1

    configured_count = int(
        metadata.get(
            "episodes_per_scene_per_level_test_type",
            metadata.get(
                "episodes_per_scene_per_level",
                metadata.get("episodes_per_scene", 0),
            ),
        )
    )
    configured_levels = metadata.get("difficulty_levels") or [None]
    configured_test_types = metadata.get("test_types") or [None]
    for scenario in metadata.get("scenarios", []):
        for difficulty_level in configured_levels:
            for test_type in configured_test_types:
                group_parts = [str(scenario)]
                if difficulty_level is not None:
                    group_parts.append(f"level{difficulty_level}")
                if test_type is not None:
                    group_parts.append(str(test_type))
                group = "/".join(group_parts)
                if counts[group] != configured_count:
                    errors.append(
                        f"{group}: expected {configured_count} episodes, "
                        f"found {counts[group]}"
                    )
                values = list(answer_positions[group].values())
                if values and max(values) - min(values) > 1:
                    errors.append(f"{group}: answer positions are not balanced")
                if task_modes.get(group) == "single_binary":
                    values = list(target_presence[group].values())
                    if len(values) != 2 or max(values) - min(values) > 1:
                        errors.append(f"{group}: binary conditions are not balanced")
                elif task_modes.get(group) == "multi_arm":
                    values = list(target_positions[group].values())
                    if values and max(values) - min(values) > 1:
                        errors.append(f"{group}: target positions are not balanced")

                values = list(mapping_options[group].values())
                if values and max(values) - min(values) > 1:
                    errors.append(f"{group}: mapping conditions are not balanced")
                values = list(behavior_positions[group].values())
                if task_modes.get(group) == "multi_arm" and values:
                    if max(values) - min(values) > 1:
                        errors.append(f"{group}: behavior positions are not balanced")
                configured_templates = metadata.get("environment_templates", [])
                if configured_templates:
                    template_counts = environment_template_counts[group]
                    if set(template_counts) != set(configured_templates) or len(
                        set(template_counts.values())
                    ) != 1:
                        errors.append(
                            f"{group}: environment templates are not balanced"
                        )

    if metadata.get("paired_nuisance", False):
        expected_pair_size = int(metadata.get("nuisance_pair_size", 0))
        if expected_pair_size != 2:
            errors.append("paired nuisance datasets must use nuisance_pair_size=2")
        for key, signatures_for_pair in nuisance_pair_signatures.items():
            if len(signatures_for_pair) != 1:
                errors.append(
                    f"{key[0]}/{key[1]}: nuisance signature differs across "
                    "conditions, difficulty levels, or test types"
                )
        for key, templates_for_pair in nuisance_pair_templates.items():
            if len(templates_for_pair) != 1:
                errors.append(
                    f"{key[0]}/{key[1]}: environment template differs "
                    "across conditions, difficulty levels, or test types"
                )
        for key, condition_counts in nuisance_pair_conditions.items():
            member_count = sum(condition_counts.values())
            if member_count != expected_pair_size:
                errors.append(
                    f"{key[0]}/level{key[1]}/{key[2]}/{key[3]}: expected "
                    f"{expected_pair_size} paired episodes, found {member_count}"
                )
            if len(condition_counts) != expected_pair_size or any(
                count != 1 for count in condition_counts.values()
            ):
                errors.append(
                    f"{key[0]}/level{key[1]}/{key[2]}/{key[3]}: paired conditions "
                    "are not one-to-one"
                )
        configured_levels = metadata.get("difficulty_levels", [])
        configured_test_types = metadata.get("test_types") or [None]
        for scenario in metadata.get("scenarios", []):
            expected_pair_ids = None
            for difficulty_level in configured_levels:
                for test_type in configured_test_types:
                    pair_ids = nuisance_pair_ids_by_group[
                        (scenario, difficulty_level, test_type)
                    ]
                    if expected_pair_ids is None:
                        expected_pair_ids = pair_ids
                    elif pair_ids != expected_pair_ids:
                        errors.append(
                            f"{scenario}: nuisance pair IDs differ across "
                            "difficulty levels or test types"
                        )
                        break

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
        "nuisance_pair_groups": len(nuisance_pair_conditions),
        "environment_templates": {
            group: dict(counter)
            for group, counter in environment_template_counts.items()
        },
        "errors": errors,
        "warnings": warnings,
    }
