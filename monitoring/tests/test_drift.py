import shutil
import unittest
from pathlib import Path

import pandas as pd

from monitoring.ledgerml_monitoring.drift import build_baseline, detect_drift


class DriftTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("monitoring/tests/.artifacts")
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)
        self.baseline_path = self.root / "baseline.csv"
        self.current_path = self.root / "current.csv"
        pd.DataFrame({"amount": range(1, 101), "velocity": [1] * 100}).to_csv(
            self.baseline_path, index=False
        )

    def tearDown(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)

    def test_drift_cases_are_table_driven(self) -> None:
        cases = [
            (
                "within baseline",
                pd.DataFrame({"amount": range(1, 101), "velocity": [1] * 100}),
                "within_baseline",
                0,
            ),
            (
                "distribution shift",
                pd.DataFrame({"amount": range(1000, 1100), "velocity": [1] * 100}),
                "drift_detected",
                1,
            ),
            (
                "missing column",
                pd.DataFrame({"velocity": [1] * 100}),
                "drift_detected",
                1,
            ),
        ]
        baseline = build_baseline(
            self.baseline_path,
            model_version="model-v1",
            dataset_version="dataset-v1",
        )
        for name, current, expected_status, minimum_signals in cases:
            with self.subTest(case=name):
                current.to_csv(self.current_path, index=False)
                report = detect_drift(baseline, self.current_path)
                self.assertEqual(report["status"], expected_status)
                self.assertGreaterEqual(len(report["signals"]), minimum_signals)

    def test_missing_rate_is_reported(self) -> None:
        baseline = build_baseline(
            self.baseline_path,
            model_version="model-v1",
            dataset_version="dataset-v1",
        )
        pd.DataFrame({"amount": [None] * 50 + list(range(1, 51)), "velocity": [1] * 100}).to_csv(
            self.current_path, index=False
        )
        report = detect_drift(baseline, self.current_path)
        self.assertTrue(any(signal["metric"] == "missing_rate_delta" for signal in report["signals"]))


if __name__ == "__main__":
    unittest.main()
