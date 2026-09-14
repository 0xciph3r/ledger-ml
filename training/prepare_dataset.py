"""CLI for preparing a validated double-entry snapshot."""

from __future__ import annotations

import argparse

from ledgerml_training.dataset_preparation import prepare_double_entry_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, help="Input snapshot directory")
    parser.add_argument("--output", required=True, help="Curated dataset directory")
    parser.add_argument("--source-version", required=True, help="Immutable source version")
    args = parser.parse_args()

    result = prepare_double_entry_dataset(
        snapshot_path=args.snapshot,
        output_path=args.output,
        source_dataset_version=args.source_version,
    )
    print(result["manifest_path"])


if __name__ == "__main__":
    main()
