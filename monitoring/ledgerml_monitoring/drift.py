"""Baseline and drift detection for curated numeric feature windows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DRIFT_BASELINE_SCHEMA_VERSION = "drift-baseline-v1"


def _numeric_series(frame: pd.DataFrame, feature: str) -> pd.Series:
    return pd.to_numeric(frame[feature], errors="coerce")


def _histogram(values: np.ndarray, edges: np.ndarray) -> list[float]:
    counts, _ = np.histogram(values, bins=edges)
    total = max(int(counts.sum()), 1)
    return (counts / total).astype(float).tolist()


def build_baseline(
    features_path: str | Path,
    *,
    model_version: str,
    dataset_version: str,
    bin_count: int = 10,
) -> dict[str, Any]:
    """Build a reproducible numeric feature baseline from a CSV window."""
    if bin_count < 2:
        raise ValueError("bin_count must be >= 2")
    frame = pd.read_csv(features_path)
    features: dict[str, Any] = {}
    for column in frame.columns:
        values = _numeric_series(frame, column)
        observed = values.dropna().to_numpy(dtype=float)
        if len(observed) == 0:
            continue
        quantiles = np.linspace(0.0, 1.0, bin_count + 1)
        edges = np.unique(np.quantile(observed, quantiles)).astype(float)
        if len(edges) < 2:
            edges = np.array([observed[0] - 0.5, observed[0] + 0.5])
        else:
            edges[0] = -np.inf
            edges[-1] = np.inf
        features[column] = {
            "count": int(len(frame)),
            "missing_rate": float(values.isna().mean()),
            "bin_edges": edges.tolist(),
            "proportions": _histogram(observed, edges),
        }
    return {
        "schema_version": DRIFT_BASELINE_SCHEMA_VERSION,
        "model_version": model_version,
        "dataset_version": dataset_version,
        "row_count": int(len(frame)),
        "features": features,
    }


def _psi(expected: list[float], actual: list[float]) -> float:
    expected_values = np.clip(np.asarray(expected, dtype=float), 1e-6, None)
    actual_values = np.clip(np.asarray(actual, dtype=float), 1e-6, None)
    return float(np.sum((actual_values - expected_values) * np.log(actual_values / expected_values)))


def detect_drift(
    baseline: dict[str, Any],
    current_features_path: str | Path,
    *,
    psi_threshold: float = 0.2,
    missing_rate_delta_threshold: float = 0.1,
) -> dict[str, Any]:
    """Compare a current feature window against a stored baseline."""
    if psi_threshold < 0.0 or missing_rate_delta_threshold < 0.0:
        raise ValueError("drift thresholds must be >= 0")
    frame = pd.read_csv(current_features_path)
    signals: list[dict[str, Any]] = []
    for feature, expected in baseline.get("features", {}).items():
        if feature not in frame.columns:
            signals.append({"feature": feature, "metric": "missing_column", "severity": "high"})
            continue
        values = _numeric_series(frame, feature)
        observed = values.dropna().to_numpy(dtype=float)
        current_missing_rate = float(values.isna().mean())
        missing_delta = current_missing_rate - float(expected["missing_rate"])
        if missing_delta > missing_rate_delta_threshold:
            signals.append({
                "feature": feature,
                "metric": "missing_rate_delta",
                "value": missing_delta,
                "threshold": missing_rate_delta_threshold,
                "severity": "high",
            })
        if len(observed) == 0:
            continue
        edges = np.asarray(expected["bin_edges"], dtype=float)
        actual_proportions = _histogram(observed, edges)
        psi = _psi(expected["proportions"], actual_proportions)
        if psi >= psi_threshold:
            signals.append({
                "feature": feature,
                "metric": "psi",
                "value": psi,
                "threshold": psi_threshold,
                "severity": "high" if psi >= psi_threshold * 2 else "medium",
            })
    return {
        "schema_version": "drift-report-v1",
        "model_version": baseline.get("model_version", "unknown"),
        "baseline_dataset_version": baseline.get("dataset_version", "unknown"),
        "current_row_count": int(len(frame)),
        "status": "drift_detected" if signals else "within_baseline",
        "signals": signals,
    }


def write_json(payload: dict[str, Any], output_path: str | Path) -> None:
    Path(output_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
