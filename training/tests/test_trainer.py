import json
import shutil
import unittest
from pathlib import Path

from training.ledgerml_training.double_entry_snapshot import DOUBLE_ENTRY_FEATURE_NAMES
from training.ledgerml_training.trainer import (
    FEATURE_NAMES,
    ConfigurationError,
    generate_synthetic_transactions,
    load_training_config,
    run_training,
)


def valid_env(output_root: Path) -> dict[str, str]:
    return {
        "LEDGERML_TASK": "fraud-scoring",
        "LEDGERML_DATASET_KIND": "LocalPath",
        "LEDGERML_DATASET_NAME": "synthetic",
        "LEDGERML_DATASET_PATH": "profiles/default",
        "LEDGERML_DATASET_VERSION": "synthetic-fraud-2026-09",
        "LEDGERML_OUTPUT_KIND": "LocalPath",
        "LEDGERML_OUTPUT_NAME": str(output_root),
        "LEDGERML_OUTPUT_PATH": "runs",
        "LEDGERML_OUTPUT_ARTIFACT_VERSION": "model-v1",
        "LEDGERML_TRAINING_IMAGE_DIGEST": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "LEDGERML_CONFIGURATION_DIGEST": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "LEDGERML_RANDOM_SEED": "1234",
        "LEDGERML_NUM_SAMPLES": "5000",
        "LEDGERML_TEST_SIZE": "0.2",
        "LEDGERML_CLASSIFICATION_THRESHOLD": "0.15",
    }


class TrainerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = Path("training/tests/.artifacts")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(parents=True)

    def tearDown(self) -> None:
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_missing_required_env_fails(self) -> None:
        env = valid_env(self.test_dir)
        env.pop("LEDGERML_DATASET_VERSION")
        with self.assertRaises(ConfigurationError):
            load_training_config(env)

    def test_deterministic_data_generation(self) -> None:
        config = load_training_config(valid_env(self.test_dir))
        features_a, labels_a, _ = generate_synthetic_transactions(config)
        features_b, labels_b, _ = generate_synthetic_transactions(config)

        self.assertTrue(features_a.equals(features_b))
        self.assertListEqual(labels_a.tolist(), labels_b.tolist())

    def test_features_and_imbalance(self) -> None:
        config = load_training_config(valid_env(self.test_dir))
        features, labels, _ = generate_synthetic_transactions(config)

        self.assertEqual(list(features.columns), list(FEATURE_NAMES))
        fraud_rate = float(labels.mean())
        self.assertGreater(fraud_rate, 0.01)
        self.assertLess(fraud_rate, 0.15)

    def test_training_outputs_metrics_and_artifacts(self) -> None:
        config = load_training_config(valid_env(self.test_dir))
        outputs = run_training(config)

        model_path = outputs["model_artifact_path"]
        evaluation_path = outputs["evaluation_path"]
        self.assertTrue(model_path.exists(), "model artifact should exist")
        self.assertTrue(evaluation_path.exists(), "evaluation json should exist")

        with evaluation_path.open("r", encoding="utf-8") as f:
            evaluation = json.load(f)

        metrics = evaluation["metrics"]
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)
        self.assertIn("pr_auc", metrics)
        self.assertIn("roc_auc", metrics)
        self.assertGreater(metrics["precision"], 0.0)
        self.assertGreater(metrics["recall"], 0.0)
        self.assertGreater(metrics["f1"], 0.0)

        confusion = evaluation["confusion_matrix"]
        for key in ("tn", "fp", "fn", "tp"):
            self.assertIn(key, confusion)
            self.assertGreaterEqual(int(confusion[key]), 0)

        self.assertEqual(evaluation["lineage"]["feature_names"], list(FEATURE_NAMES))
        self.assertEqual(evaluation["dataset"]["kind"], "LocalPath")
        self.assertEqual(evaluation["dataset"]["schema_version"], "synthetic-fraud-v1")
        self.assertIn("Synthetic fraud labels", evaluation["dataset"]["label_caveat"])

    def test_double_entry_snapshot_dataset_training_metadata(self) -> None:
        fixture_path = Path("training/tests/fixtures/double_entry_snapshot")
        env = valid_env(self.test_dir)
        env.update(
            {
                "LEDGERML_DATASET_KIND": "DoubleEntryLedgerSnapshot",
                "LEDGERML_DATASET_NAME": "double-entry-sanitized-fixture",
                "LEDGERML_DATASET_PATH": str(fixture_path),
                "LEDGERML_DATASET_VERSION": "double-entry-snapshot-v1",
                "LEDGERML_OUTPUT_ARTIFACT_VERSION": "double-entry-model-v1",
            }
        )
        config = load_training_config(env)
        outputs = run_training(config)

        with outputs["evaluation_path"].open("r", encoding="utf-8") as f:
            evaluation = json.load(f)

        self.assertEqual(evaluation["dataset"]["kind"], "DoubleEntryLedgerSnapshot")
        self.assertEqual(
            evaluation["dataset"]["schema_version"], "double-entry-ledger-snapshot-v1"
        )
        self.assertIn("proxy", evaluation["dataset"]["label_definition"].lower())
        self.assertIn("not confirmed fraud ground truth", evaluation["dataset"]["label_caveat"])
        self.assertEqual(
            evaluation["lineage"]["feature_names"], list(DOUBLE_ENTRY_FEATURE_NAMES)
        )


if __name__ == "__main__":
    unittest.main()
