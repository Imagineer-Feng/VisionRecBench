import argparse
import json
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from source.dataset_io import validate_dataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Validate a frozen VisionRecBench dataset and its balance."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--skip-checksums",
        action="store_true",
        help="Check structure and image readability without hashing every image.",
    )
    args = parser.parse_args()
    report = validate_dataset(
        args.dataset.resolve(),
        verify_checksums=not args.skip_checksums,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
