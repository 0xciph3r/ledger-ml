"""CLI for evaluating delayed model outcomes."""

from __future__ import annotations

import argparse

from ledgerml_monitoring.outcomes import evaluate_delayed_outcomes, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--outcomes", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = evaluate_delayed_outcomes(
        args.predictions,
        args.outcomes,
        model_version=args.model_version,
    )
    write_report(report, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
