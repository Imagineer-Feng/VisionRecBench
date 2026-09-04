import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from source.agent import EVALUATION_PROTOCOL_VERSION  # noqa: E402
from source.dataset_io import (  # noqa: E402
    atomic_write_json,
    load_json,
    load_manifest,
    utc_timestamp,
)
from source.logic_audit import LOGIC_AUDIT_PROTOCOL_VERSION  # noqa: E402
from source.logic_audit_metrics import summarize_logic_audit  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Recalculate accuracy after requiring every initially correct answer "
            "to pass its exact structured logic audit."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-results", type=Path, required=True)
    parser.add_argument("--audit-results", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _load_base_results(root, model, dataset_name, dataset_hash):
    rows = []
    for path in Path(root).glob("**/*.json"):
        try:
            row = load_json(path)
        except Exception:
            continue
        required = {
            "dataset_name",
            "dataset_content_sha256",
            "episode_id",
            "model",
            "evaluation_protocol_version",
            "difficulty_level",
            "scenario",
            "test_type",
            "correct",
        }
        if (
            required.issubset(row)
            and row["model"] == model
            and row["dataset_name"] == dataset_name
            and row["dataset_content_sha256"] == dataset_hash
            and row["evaluation_protocol_version"]
            == EVALUATION_PROTOCOL_VERSION
        ):
            rows.append(row)
    return rows


def _load_audit_results(root, model, dataset_name, dataset_hash):
    rows = []
    for path in Path(root).glob("**/*.json"):
        try:
            row = load_json(path)
        except Exception:
            continue
        required = {
            "dataset_name",
            "dataset_content_sha256",
            "episode_id",
            "model",
            "base_evaluation_protocol_version",
            "logic_audit_protocol_version",
            "difficulty_level",
            "scenario",
            "test_type",
            "logic_audit_pass",
        }
        if (
            required.issubset(row)
            and row["model"] == model
            and row["dataset_name"] == dataset_name
            and row["dataset_content_sha256"] == dataset_hash
            and row["base_evaluation_protocol_version"]
            == EVALUATION_PROTOCOL_VERSION
            and row["logic_audit_protocol_version"]
            == LOGIC_AUDIT_PROTOCOL_VERSION
        ):
            rows.append(row)
    return rows


def _group_key(row, protocol_field):
    return (
        row["dataset_name"],
        row["dataset_content_sha256"],
        row["model"],
        row[protocol_field],
        int(row["difficulty_level"]),
        row["scenario"],
        row["test_type"],
    )


def main():
    args = parse_args()
    metadata = load_json(args.dataset / "metadata.json")
    dataset_name = metadata["dataset_name"]
    dataset_hash = metadata["content_sha256"]
    manifest_ids = {
        row["episode_id"] for row in load_manifest(args.dataset)
    }
    base_rows = _load_base_results(
        args.base_results,
        args.model,
        dataset_name,
        dataset_hash,
    )
    if not base_rows:
        raise SystemExit(
            f"No {EVALUATION_PROTOCOL_VERSION} ordinary results for {args.model} "
            f"under {args.base_results}"
        )
    base_ids = [row["episode_id"] for row in base_rows]
    if len(base_ids) != len(set(base_ids)):
        raise SystemExit("Duplicate ordinary result files were found.")
    missing = sorted(manifest_ids - set(base_ids))
    extra = sorted(set(base_ids) - manifest_ids)
    if missing or extra:
        detail = f"missing={len(missing)}, extra={len(extra)}"
        if missing:
            detail += f", first_missing={missing[0]}"
        raise SystemExit(
            "Ordinary evaluation is not complete for this frozen dataset "
            f"({detail})."
        )
    audit_rows = _load_audit_results(
        args.audit_results,
        args.model,
        dataset_name,
        dataset_hash,
    )

    base_groups = defaultdict(list)
    for row in base_rows:
        base_groups[_group_key(row, "evaluation_protocol_version")].append(row)
    audit_groups = defaultdict(list)
    for row in audit_rows:
        audit_groups[_group_key(row, "base_evaluation_protocol_version")].append(
            row
        )

    unknown_groups = sorted(set(audit_groups) - set(base_groups))
    if unknown_groups:
        raise SystemExit(
            "Found logic-audit groups without matching ordinary results: "
            f"{unknown_groups[:3]}"
        )

    groups = []
    for key, group_base_rows in sorted(base_groups.items()):
        metrics = summarize_logic_audit(
            group_base_rows,
            audit_groups.get(key, []),
        )
        groups.append(
            {
                "dataset_name": key[0],
                "dataset_content_sha256": key[1],
                "model": key[2],
                "base_evaluation_protocol_version": key[3],
                "logic_audit_protocol_version": LOGIC_AUDIT_PROTOCOL_VERSION,
                "difficulty_level": key[4],
                "scenario": key[5],
                "test_type": key[6],
                **metrics,
            }
        )

    overall_groups = defaultdict(lambda: {"base": [], "audit": []})
    for key, group_base_rows in base_groups.items():
        overall_key = key[:4]
        overall_groups[overall_key]["base"].extend(group_base_rows)
        overall_groups[overall_key]["audit"].extend(audit_groups.get(key, []))
    overall = []
    for key, rows in sorted(overall_groups.items()):
        overall.append(
            {
                "dataset_name": key[0],
                "dataset_content_sha256": key[1],
                "model": key[2],
                "base_evaluation_protocol_version": key[3],
                "logic_audit_protocol_version": LOGIC_AUDIT_PROTOCOL_VERSION,
                **summarize_logic_audit(rows["base"], rows["audit"]),
            }
        )

    report = {
        "created_at": utc_timestamp(),
        "groups": groups,
        "overall": overall,
    }
    if args.output is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        atomic_write_json(args.output, report)
        print(f"logic-audit summary saved to {args.output}")


if __name__ == "__main__":
    main()
