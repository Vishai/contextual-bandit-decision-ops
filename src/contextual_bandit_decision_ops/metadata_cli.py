from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import PackageNotFoundError, version

PROJECT_NAME = "contextual-bandit-decision-ops"
SOURCE_VERSION = "0.1.0"


def project_version() -> str:
    try:
        return version(PROJECT_NAME)
    except PackageNotFoundError:
        return SOURCE_VERSION


def project_metadata() -> dict[str, object]:
    return {
        "name": PROJECT_NAME,
        "version": project_version(),
        "service_mode": "local-staging-only",
        "python_requires": ">=3.11",
        "cpu_only": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show project release metadata")
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print only the installed package version",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.version:
        print(project_version())
    else:
        print(json.dumps(project_metadata(), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
