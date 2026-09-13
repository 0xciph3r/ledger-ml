import shutil
import unittest
from pathlib import Path

import pandas as pd

from training.ledgerml_training.double_entry_snapshot import (
    DOUBLE_ENTRY_FEATURE_NAMES,
    SnapshotContractError,
    load_double_entry_snapshot,
)


FIXTURE_ROOT = Path("training/tests/fixtures/double_entry_snapshot")


class DoubleEntrySnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.test_root = Path("training/tests/.artifacts/double_entry_snapshot")
        if self.test_root.exists():
            shutil.rmtree(self.test_root)
        self.test_root.mkdir(parents=True)

    def tearDown(self) -> None:
        if self.test_root.exists():
            shutil.rmtree(self.test_root)

    def _copy_fixture(self, target_name: str) -> Path:
        target = self.test_root / target_name
        shutil.copytree(FIXTURE_ROOT, target)
        return target

    def test_balanced_snapshot_loads(self) -> None:
        snapshot_path = self._copy_fixture("balanced")
        dataset = load_double_entry_snapshot(str(snapshot_path))

        self.assertEqual(list(dataset.features.columns), list(DOUBLE_ENTRY_FEATURE_NAMES))
        self.assertEqual(len(dataset.features), 12)
        self.assertEqual(dataset.metadata["schema_version"], "double-entry-ledger-snapshot-v1")
        self.assertEqual(dataset.metadata["transaction_ids_order"][0], "tx001")
        self.assertEqual(dataset.metadata["transaction_ids_order"][-1], "tx012")

    def test_unbalanced_transaction_is_rejected(self) -> None:
        snapshot_path = self._copy_fixture("unbalanced")
        entries = pd.read_csv(snapshot_path / "entries.csv", dtype=str)
        entries.loc[entries["entry_id"] == "e004", "amount_base_units"] = "19999"
        entries.to_csv(snapshot_path / "entries.csv", index=False)

        with self.assertRaises(SnapshotContractError) as ctx:
            load_double_entry_snapshot(str(snapshot_path))
        self.assertIn("unbalanced", str(ctx.exception).lower())

    def test_invalid_decimal_and_negative_amounts_rejected(self) -> None:
        for case_name, replacement in (("decimal", "12.50"), ("negative", "-50"), ("invalid", "abc")):
            with self.subTest(case=case_name):
                snapshot_path = self._copy_fixture(case_name)
                entries = pd.read_csv(snapshot_path / "entries.csv", dtype=str)
                entries.loc[entries["entry_id"] == "e001", "amount_base_units"] = replacement
                entries.to_csv(snapshot_path / "entries.csv", index=False)

                with self.assertRaises(SnapshotContractError):
                    load_double_entry_snapshot(str(snapshot_path))

    def test_proxy_labels_and_leakage_boundaries(self) -> None:
        snapshot_path = self._copy_fixture("labels")
        dataset = load_double_entry_snapshot(str(snapshot_path))

        self.assertListEqual(
            dataset.labels.tolist(),
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
        )
        forbidden = {
            "settlement_state",
            "reversal_id",
            "reversed_at",
            "reason_code",
            "proxy_label",
            "label",
        }
        self.assertTrue(forbidden.isdisjoint(set(dataset.features.columns)))
        self.assertIn("Proxy risk labels", dataset.metadata["label_caveat"])

    def test_integer_money_and_temporal_velocity_features(self) -> None:
        snapshot_path = self._copy_fixture("velocity")
        dataset = load_double_entry_snapshot(str(snapshot_path))

        first_row = dataset.features.iloc[0]
        self.assertEqual(int(first_row["total_amount_base_units"]), 12500)
        self.assertEqual(int(first_row["entry_count"]), 2)
        self.assertEqual(int(first_row["debit_entry_count"]), 1)
        self.assertEqual(int(first_row["credit_entry_count"]), 1)

        tx002_row = dataset.features.iloc[1]
        self.assertEqual(int(tx002_row["prior_account_tx_count_1h"]), 2)
        self.assertEqual(int(tx002_row["prior_account_tx_count_24h"]), 2)

        tx006_row = dataset.features.iloc[5]
        self.assertEqual(int(tx006_row["prior_account_tx_count_1h"]), 2)
        self.assertEqual(int(tx006_row["prior_account_tx_count_24h"]), 9)

    def test_mixed_currency_rejected(self) -> None:
        snapshot_path = self._copy_fixture("mixed_currency")
        entries = pd.read_csv(snapshot_path / "entries.csv", dtype=str)
        entries.loc[entries["entry_id"] == "e002", "currency"] = "EUR"
        entries.to_csv(snapshot_path / "entries.csv", index=False)

        with self.assertRaises(SnapshotContractError) as ctx:
            load_double_entry_snapshot(str(snapshot_path))
        self.assertIn("currency", str(ctx.exception).lower())

    def test_deterministic_feature_extraction(self) -> None:
        snapshot_path = self._copy_fixture("deterministic")
        dataset_a = load_double_entry_snapshot(str(snapshot_path))
        dataset_b = load_double_entry_snapshot(str(snapshot_path))

        self.assertTrue(dataset_a.features.equals(dataset_b.features))
        self.assertListEqual(dataset_a.labels.tolist(), dataset_b.labels.tolist())


if __name__ == "__main__":
    unittest.main()
