import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from source.dataset_io import atomic_write_json, load_json, utc_timestamp  # noqa: E402
from source.offline_metrics import summarize_result_group  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize frozen-dataset results with episode-level bootstrap intervals."
        )
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.bootstrap_samples < 0:
        parser.error("--bootstrap-samples must be non-negative")
    return args


def load_offline_results(root):
    rows = []
    for path in Path(root).glob("**/*.json"):
        try:
            row = load_json(path)
        except Exception:
            continue
        required = {
            "dataset_content_sha256",
            "episode_id",
            "model",
            "scenario",
            "correct",
        }
        if required.issubset(row) and "difficulty_level" in row:
            rows.append(row)
    return rows


def main():
    args = parse_args()
    rows = load_offline_results(args.input)
    if not rows:
        raise SystemExit(f"No offline result JSON files found under {args.input}")

    groups = defaultdict(list)
    for row in rows:
        key = (
            row["dataset_name"],
            row["dataset_content_sha256"],
            row["model"],
            int(row["difficulty_level"]),
            row["scenario"],
        )
        groups[key].append(row)

    summaries = []
    for group_index, (key, group_rows) in enumerate(sorted(groups.items())):
        dataset_name, dataset_hash, model, level, scenario = key
        metrics = summarize_result_group(
            group_rows,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + group_index,
        )
        summaries.append(
            {
                "dataset_name": dataset_name,
                "dataset_content_sha256": dataset_hash,
                "model": model,
                "difficulty_level": level,
                "scenario": scenario,
                **metrics,
            }
        )

    macro_groups = defaultdict(list)
    for summary in summaries:
        key = (
            summary["dataset_name"],
            summary["dataset_content_sha256"],
            summary["model"],
            summary["difficulty_level"],
        )
        primary_score = summary.get(
            "balanced_accuracy",
            summary.get("macro_position_recall", summary["accuracy"]),
        )
        macro_groups[key].append(primary_score)
    macro_summaries = [
        {
            "dataset_name": key[0],
            "dataset_content_sha256": key[1],
            "model": key[2],
            "difficulty_level": key[3],
            "scene_macro_score": round(sum(values) / len(values), 6),
            "scenes": len(values),
        }
        for key, values in sorted(macro_groups.items())
    ]

    report = {
        "created_at": utc_timestamp(),
        "bootstrap_samples": args.bootstrap_samples,
        "groups": summaries,
        "macro": macro_summaries,
    }
    if args.output is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        atomic_write_json(args.output, report)
        print(f"summary saved to {args.output}")


if __name__ == "__main__":
    main()
