"""CLI for building a drift baseline from curated feature data."""

from __future__ import annotations

import argparse

from ledgerml_monitoring.drift import build_baseline, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline = build_baseline(
        args.features,
        model_version=args.model_version,
        dataset_version=args.dataset_version,
    )
    write_json(baseline, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
