import shutil
import unittest
from pathlib import Path

import pandas as pd

from monitoring.ledgerml_monitoring.outcomes import (
    OutcomeContractError,
    evaluate_delayed_outcomes,
)


class OutcomeMonitoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("monitoring/tests/.artifacts/outcomes")
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)
        self.predictions = self.root / "predictions.csv"
        self.outcomes = self.root / "outcomes.csv"
        pd.DataFrame(
            {
                "prediction_id": ["p1", "p2", "p3", "p4"],
                "transaction_id": ["tx1", "tx2", "tx3", "tx4"],
                "model_version": ["model-v1"] * 4,
                "predicted_label": ["1", "0", "1", "0"],
                "risk_score": ["0.9", "0.1", "0.8", "0.2"],
            }
        ).to_csv(self.predictions, index=False)

    def tearDown(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)

    def test_outcome_contract_is_table_driven(self) -> None:
        cases = [
            (
                "complete outcomes",
                pd.DataFrame(
                    {
                        "transaction_id": ["tx1", "tx2", "tx3", "tx4"],
                        "outcome_label": ["1", "0", "0", "1"],
                        "observed_at": ["2026-09-14T10:00:00Z"] * 4,
                    }
                ),
                None,
                4,
            ),
            (
                "delayed partial outcomes",
                pd.DataFrame(
                    {
                        "transaction_id": ["tx1", "tx2"],
                        "outcome_label": ["1", "0"],
                        "observed_at": ["2026-09-14T10:00:00Z"] * 2,
                    }
                ),
                None,
                2,
            ),
            (
                "invalid label",
                pd.DataFrame(
                    {
                        "transaction_id": ["tx1"],
                        "outcome_label": ["fraud"],
                        "observed_at": ["2026-09-14T10:00:00Z"],
                    }
                ),
                OutcomeContractError,
                None,
            ),
        ]
        for name, outcomes, expected_error, expected_matches in cases:
            with self.subTest(case=name):
                outcomes.to_csv(self.outcomes, index=False)
                if expected_error:
                    with self.assertRaises(expected_error):
                        evaluate_delayed_outcomes(self.predictions, self.outcomes, model_version="model-v1")
                else:
                    report = evaluate_delayed_outcomes(
                        self.predictions, self.outcomes, model_version="model-v1"
                    )
                    self.assertEqual(report["matched_outcome_count"], expected_matches)
                    self.assertEqual(report["model_version"], "model-v1")

    def test_duplicate_outcomes_fail_closed(self) -> None:
        pd.DataFrame(
            {
                "transaction_id": ["tx1", "tx1"],
                "outcome_label": ["1", "0"],
                "observed_at": ["2026-09-14T10:00:00Z"] * 2,
            }
        ).to_csv(self.outcomes, index=False)
        with self.assertRaisesRegex(OutcomeContractError, "duplicate"):
            evaluate_delayed_outcomes(self.predictions, self.outcomes, model_version="model-v1")


if __name__ == "__main__":
    unittest.main()
