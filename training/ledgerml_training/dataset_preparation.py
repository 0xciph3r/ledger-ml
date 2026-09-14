"""Prepare validated ledger snapshots into curated training datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .double_entry_snapshot import (
    DOUBLE_ENTRY_FEATURE_NAMES,
    SnapshotContractError,
    _resolve_snapshot_path,
    load_double_entry_snapshot,
)


CURATED_DATASET_SCHEMA_VERSION = "curated-ledger-dataset-v1"
MANIFEST_FILENAME = "dataset-manifest.json"
FEATURES_FILENAME = "features.csv"
LABELS_FILENAME = "labels.csv"


def _sha256_files(root: Path, filenames: list[str]) -> str:
    digest = hashlib.sha256()
    for filename in sorted(filenames):
        path = root / filename
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def _snapshot_digest(snapshot_root: Path) -> str:
    filenames = [
        path.name
        for path in snapshot_root.iterdir()
        if path.is_file() and path.suffix.lower() == ".csv"
    ]
    return _sha256_files(snapshot_root, filenames)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def prepare_double_entry_dataset(
    snapshot_path: str,
    output_path: str,
    source_dataset_version: str,
) -> dict[str, Any]:
    """Validate a snapshot and write a deterministic curated dataset.

    The labels are written separately from the feature matrix so downstream
    training code has an explicit opportunity to enforce the leakage boundary.
    """
    if not source_dataset_version.strip():
        raise ValueError("source_dataset_version must not be empty")

    snapshot_root = _resolve_snapshot_path(snapshot_path)
    output_root = Path(output_path).expanduser()
    if not output_root.is_absolute():
        output_root = Path.cwd() / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        dataset = load_double_entry_snapshot(str(snapshot_root))
    except SnapshotContractError:
        raise

    transaction_ids = dataset.metadata["transaction_ids_order"]
    features = dataset.features.copy()
    features.insert(0, "transaction_id", transaction_ids)
    labels = pd.DataFrame(
        {"transaction_id": transaction_ids, "label": dataset.labels.astype(int)}
    )

    features_path = output_root / FEATURES_FILENAME
    labels_path = output_root / LABELS_FILENAME
    _write_csv(features, features_path)
    _write_csv(labels, labels_path)

    timestamps = [
        value
        for value in dataset.metadata.get("transaction_created_at", [])
        if value
    ]
    manifest: dict[str, Any] = {
        "schema_version": CURATED_DATASET_SCHEMA_VERSION,
        "source": {
            "kind": "DoubleEntryLedgerSnapshot",
            "version": source_dataset_version,
            "snapshot_digest": _snapshot_digest(snapshot_root),
            "point_in_time_cutoff": max(timestamps) if timestamps else None,
        },
        "features": {
            "schema_version": dataset.metadata["schema_version"],
            "feature_names": list(DOUBLE_ENTRY_FEATURE_NAMES),
            "label_column": "label",
            "leakage_boundary": "labels.csv is separate from features.csv; outcome fields are not features",
        },
        "label_definition": dataset.metadata["label_definition"],
        "label_caveat": dataset.metadata["label_caveat"],
        "quality": {
            "row_count": int(len(features)),
            "positive_label_count": int(dataset.labels.sum()),
            "rejected_record_count": 0,
            "double_entry_validation": "passed",
        },
        "files": {
            "features": FEATURES_FILENAME,
            "labels": LABELS_FILENAME,
        },
    }
    manifest["content_digest"] = _sha256_files(
        output_root, [FEATURES_FILENAME, LABELS_FILENAME]
    )
    manifest_path = output_root / MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "features_path": features_path,
        "labels_path": labels_path,
        "manifest_path": manifest_path,
        "manifest": manifest,
    }
