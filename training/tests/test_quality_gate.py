import json
import shutil
import unittest
from pathlib import Path

from training.ledgerml_training.quality_gate import (
    EvaluationGateError,
    evaluate_quality_gate,
)


class QualityGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("training/tests/.artifacts/quality-gate")
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)
        self.path = self.root / "evaluation-lineage.json"

    def tearDown(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)

    def write_evidence(self, recall=0.82, pr_auc=0.41, false_negatives=4) -> None:
        self.path.write_text(
            json.dumps(
                {
                    "metrics": {"recall": recall, "pr_auc": pr_auc},
                    "confusion_matrix": {"fn": false_negatives},
                }
            ),
            encoding="utf-8",
        )

    def test_passes_configured_quality_gates(self) -> None:
        self.write_evidence()
        result = evaluate_quality_gate(
            self.path,
            min_recall=0.8,
            min_pr_auc=0.4,
            max_false_negatives=5,
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["false_negatives"], 4)

    def test_rejects_failed_gate(self) -> None:
        self.write_evidence(recall=0.72)
        with self.assertRaises(EvaluationGateError) as ctx:
            evaluate_quality_gate(self.path, min_recall=0.8)
        self.assertIn("recall", str(ctx.exception))

    def test_rejects_missing_required_evidence(self) -> None:
        self.path.write_text(json.dumps({"metrics": {}}), encoding="utf-8")
        with self.assertRaises(EvaluationGateError):
            evaluate_quality_gate(self.path)


if __name__ == "__main__":
    unittest.main()
