import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from scripts.evaluate_dataset import (  # noqa: E402
    build_episode_content,
    content_sha256,
    sanitized_api_endpoint,
)
from source.agent import EVALUATION_PROTOCOL_VERSION, create_identifier  # noqa: E402
from source.dataset_io import (  # noqa: E402
    DATASET_SCHEMA_VERSION,
    atomic_write_json,
    load_episode,
    load_json,
    load_manifest,
    sha256_file,
    utc_timestamp,
    validate_dataset,
)
from source.difficulty import DIFFICULTY_LEVELS  # noqa: E402
from source.logic_audit import (  # noqa: E402
    LOGIC_AUDIT_PROTOCOL_VERSION,
    LOGIC_AUDIT_SYSTEM_MESSAGE,
    build_logic_audit_prompt,
    parse_json_object,
    score_logic_audit,
)
from source.preprocess import TEST_TYPES  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Follow up only initially correct eval_v4 answers with a strict, "
            "scene-specific structured logic audit."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-results", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "results" / "logic_audit",
    )
    parser.add_argument(
        "--level",
        type=int,
        nargs="+",
        choices=DIFFICULTY_LEVELS,
        default=None,
    )
    parser.add_argument("--scenario", nargs="+", default=None)
    parser.add_argument(
        "--test-type",
        dest="test_types",
        nargs="+",
        choices=TEST_TYPES,
        default=None,
    )
    parser.add_argument(
        "--max-image-history",
        type=int,
        default=-1,
        help="Must match the setting used for the ordinary evaluation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit ordinary results examined after applying filters.",
    )
    parser.add_argument(
        "--audit-max-completion-tokens",
        type=int,
        default=8192,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-checksums", action="store_true")
    args = parser.parse_args()
    if args.model == "random":
        parser.error("The random baseline cannot perform a logic audit.")
    if args.max_image_history < -1:
        parser.error("--max-image-history must be -1 or non-negative")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.audit_max_completion_tokens < 1:
        parser.error("--audit-max-completion-tokens must be positive")
    return args


def _text_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_base_results(root, model, dataset_name, dataset_hash):
    matches = {}
    for path in Path(root).glob("**/*.json"):
        try:
            row = load_json(path)
        except Exception:
            continue
        required = {
            "dataset_name",
            "dataset_content_sha256",
            "episode_id",
            "episode_signature",
            "model",
            "evaluation_protocol_version",
            "input_content_sha256",
            "answer_index",
            "choice",
            "valid",
            "correct",
            "response_text",
        }
        if not required.issubset(row):
            continue
        if (
            row["dataset_name"] != dataset_name
            or row["dataset_content_sha256"] != dataset_hash
            or row["model"] != model
            or row["evaluation_protocol_version"] != EVALUATION_PROTOCOL_VERSION
        ):
            continue
        episode_id = row["episode_id"]
        if episode_id in matches:
            raise ValueError(f"Duplicate ordinary result for {episode_id}")
        matches[episode_id] = (path, row)
    return matches


def logic_audit_result_path(output_root, metadata, model, record):
    return (
        Path(output_root)
        / metadata["dataset_name"].replace("/", "-")
        / f"difficulty_level{record['difficulty_level']}"
        / record["test_type"]
        / model.replace("/", "-")
        / EVALUATION_PROTOCOL_VERSION
        / LOGIC_AUDIT_PROTOCOL_VERSION
        / record["scenario"]
        / f"{record['episode_id']}.json"
    )


def _select_manifest_rows(rows, args):
    if args.scenario:
        scenarios = set(args.scenario)
        rows = [row for row in rows if row["scenario"] in scenarios]
    if args.level:
        levels = set(args.level)
        rows = [
            row for row in rows
            if int(row["difficulty_level"]) in levels
        ]
    if args.test_types:
        test_types = set(args.test_types)
        rows = [row for row in rows if row["test_type"] in test_types]
    if args.limit is not None:
        rows = rows[: args.limit]
    return rows


def _validate_base_result(base, record, metadata, model, input_hash):
    expected = {
        "dataset_name": metadata["dataset_name"],
        "dataset_content_sha256": metadata["content_sha256"],
        "episode_id": record["episode_id"],
        "episode_signature": record["episode_signature"],
        "model": model,
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "input_content_sha256": input_hash,
        "answer_index": record["answer_index"],
    }
    mismatches = [
        key for key, value in expected.items() if base.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            f"Ordinary result metadata mismatch ({', '.join(mismatches)}): "
            f"{record['episode_id']}"
        )
    recomputed_correct = (
        bool(base.get("valid"))
        and base.get("choice") == int(record["answer_index"])
    )
    if bool(base.get("correct")) != recomputed_correct:
        raise ValueError(
            f"Ordinary result correctness is inconsistent: {record['episode_id']}"
        )


def _validate_existing_audit(
    existing,
    record,
    metadata,
    model,
    base_hash,
    input_hash,
    prompt_hash,
    generation_parameters,
):
    expected = {
        "dataset_content_sha256": metadata["content_sha256"],
        "episode_id": record["episode_id"],
        "episode_signature": record["episode_signature"],
        "model": model,
        "base_evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "logic_audit_protocol_version": LOGIC_AUDIT_PROTOCOL_VERSION,
        "base_result_sha256": base_hash,
        "input_content_sha256": input_hash,
        "logic_audit_prompt_sha256": prompt_hash,
        "audit_generation_parameters": generation_parameters,
    }
    mismatches = [
        key for key, value in expected.items() if existing.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            f"Existing audit metadata mismatch ({', '.join(mismatches)}): "
            f"{record['episode_id']}"
        )


def main():
    args = parse_args()
    dataset_root = args.dataset.resolve()
    validation = validate_dataset(
        dataset_root,
        verify_checksums=args.verify_checksums,
    )
    if not validation["valid"]:
        print(json.dumps(validation, indent=2, sort_keys=True))
        raise SystemExit(1)

    metadata = load_json(dataset_root / "metadata.json")
    if metadata.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise SystemExit(
            f"Logic audit requires dataset schema {DATASET_SCHEMA_VERSION}."
        )

    manifest_rows = _select_manifest_rows(load_manifest(dataset_root), args)
    base_results = load_base_results(
        args.base_results,
        args.model,
        metadata["dataset_name"],
        metadata["content_sha256"],
    )
    missing = [
        row["episode_id"]
        for row in manifest_rows
        if row["episode_id"] not in base_results
    ]
    if missing:
        raise SystemExit(
            f"Missing {len(missing)} ordinary {EVALUATION_PROTOCOL_VERSION} "
            f"results for the selected dataset; first missing: {missing[0]}"
        )

    generation_parameters = {
        "max_completion_tokens": args.audit_max_completion_tokens,
    }
    outcomes = Counter()
    identifier = None
    total = len(manifest_rows)
    for current, summary in enumerate(manifest_rows, start=1):
        base_path, base = base_results[summary["episode_id"]]
        record = load_episode(dataset_root, summary)
        outcomes["ordinary_results"] += 1

        if not base.get("correct"):
            outcomes["initially_incorrect_not_audited"] += 1
            continue

        outcomes["initially_correct"] += 1
        content_items = build_episode_content(
            record,
            args.max_image_history,
            dataset_root,
        )
        input_hash = content_sha256(content_items)
        _validate_base_result(
            base,
            record,
            metadata,
            args.model,
            input_hash,
        )
        prompt = build_logic_audit_prompt(record)
        prompt_hash = _text_sha256(prompt)
        base_hash = sha256_file(base_path)
        output_path = logic_audit_result_path(
            args.output,
            metadata,
            args.model,
            record,
        )
        if output_path.is_file():
            if not args.resume:
                raise FileExistsError(
                    f"Logic-audit result already exists: {output_path}. "
                    "Use --resume."
                )
            existing = load_json(output_path)
            _validate_existing_audit(
                existing,
                record,
                metadata,
                args.model,
                base_hash,
                input_hash,
                prompt_hash,
                generation_parameters,
            )
            outcomes["skipped"] += 1
            outcomes[
                "logic_audit_pass"
                if existing.get("logic_audit_pass")
                else "logic_audit_fail"
            ] += 1
            print(
                f"[{current}/{total}] skipping audited {record['episode_id']}",
                flush=True,
            )
            continue

        print(
            f"[{current}/{total}] auditing initially correct "
            f"{record['episode_id']}",
            flush=True,
        )
        if identifier is None:
            identifier = create_identifier(args.model)
        response_text, response_metadata = identifier.generate_follow_up(
            content_items,
            base["response_text"],
            prompt,
            LOGIC_AUDIT_SYSTEM_MESSAGE,
            generation_parameters=generation_parameters,
        )
        response_text = response_text.strip()
        score = score_logic_audit(record, response_text)
        passed = bool(score["exact_match"])
        result = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "dataset_name": metadata["dataset_name"],
            "dataset_content_sha256": metadata["content_sha256"],
            "episode_id": record["episode_id"],
            "episode_signature": record["episode_signature"],
            "scenario": record["scenario"],
            "scene": record["scene"],
            "difficulty_level": int(record["difficulty_level"]),
            "difficulty_name": record["difficulty_name"],
            "test_type": record["test_type"],
            "nuisance_pair_id": record.get("nuisance_pair_id"),
            "nuisance_signature": record.get("nuisance_signature"),
            "environment_template": record.get("environment_template"),
            "arm_type": record.get("arm_type"),
            "camera_view": record.get("camera_view"),
            "model": args.model,
            "api_endpoint": sanitized_api_endpoint(args.model),
            "base_evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
            "logic_audit_protocol_version": LOGIC_AUDIT_PROTOCOL_VERSION,
            "audit_generation_parameters": generation_parameters,
            "audited_at": utc_timestamp(),
            "base_result_sha256": base_hash,
            "input_content_sha256": input_hash,
            "initial_choice": base["choice"],
            "answer_index": record["answer_index"],
            "initial_correct": True,
            "base_response_text": base["response_text"],
            "logic_audit_system_message": LOGIC_AUDIT_SYSTEM_MESSAGE,
            "logic_audit_system_message_sha256": _text_sha256(
                LOGIC_AUDIT_SYSTEM_MESSAGE
            ),
            "logic_audit_prompt": prompt,
            "logic_audit_prompt_sha256": prompt_hash,
            "logic_audit_response_text": response_text,
            "parsed_logic_audit_response": parse_json_object(response_text),
            "logic_audit_response_id": response_metadata.get("response_id"),
            "logic_audit_response_model": response_metadata.get("response_model"),
            "logic_audit_response_created": response_metadata.get(
                "response_created"
            ),
            "logic_audit_system_fingerprint": response_metadata.get(
                "system_fingerprint"
            ),
            "logic_audit_service_tier": response_metadata.get("service_tier"),
            "logic_audit_finish_reason": response_metadata.get("finish_reason"),
            "logic_audit_token_usage": response_metadata.get("token_usage"),
            "logic_audit_score": score,
            "logic_audit_pass": passed,
            "final_correct": passed,
            "final_label": "logical_reasoning" if passed else "incorrect",
        }
        atomic_write_json(output_path, result)
        outcomes["completed"] += 1
        outcomes[
            "logic_audit_pass" if passed else "logic_audit_fail"
        ] += 1

    print(json.dumps(dict(outcomes), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
