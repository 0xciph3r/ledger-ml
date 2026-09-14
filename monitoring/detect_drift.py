"""CLI for comparing a current feature window with a drift baseline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ledgerml_monitoring.drift import detect_drift


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=os.environ.get("LEDGERML_DRIFT_BASELINE_PATH", ""))
    parser.add_argument("--current-features", default=os.environ.get("LEDGERML_DRIFT_CURRENT_FEATURES_PATH", ""))
    parser.add_argument("--output", default=os.environ.get("LEDGERML_DRIFT_REPORT_PATH", "/tmp/drift-report.json"))
    parser.add_argument("--psi-threshold", type=float, default=float(os.environ.get("LEDGERML_DRIFT_PSI_THRESHOLD", "0.2")))
    parser.add_argument("--missing-rate-delta-threshold", type=float, default=float(os.environ.get("LEDGERML_DRIFT_MISSING_RATE_DELTA_THRESHOLD", "0.1")))
    args = parser.parse_args()

    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    report = detect_drift(
        baseline,
        args.current_features,
        psi_threshold=args.psi_threshold,
        missing_rate_delta_threshold=args.missing_rate_delta_threshold,
    )
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
