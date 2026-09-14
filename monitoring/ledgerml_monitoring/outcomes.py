"""Evaluate delayed outcomes against versioned model predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class OutcomeContractError(ValueError):
    """Raised when prediction or outcome data violates the monitoring contract."""


PREDICTION_COLUMNS = (
    "prediction_id",
    "transaction_id",
    "model_version",
    "predicted_label",
    "risk_score",
)
OUTCOME_COLUMNS = ("transaction_id", "outcome_label", "observed_at")


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...], name: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise OutcomeContractError(f"{name} is missing required columns: {', '.join(missing)}")


def evaluate_delayed_outcomes(
    predictions_path: str | Path,
    outcomes_path: str | Path,
    *,
    model_version: str,
) -> dict[str, Any]:
    """Join predictions to later outcomes and compute quality metrics.

    The outcome timestamp is evidence that labels arrived after prediction. Outcome
    fields are consumed only by this monitoring stage and never become serving
    features.
    """
    predictions = pd.read_csv(predictions_path, dtype=str)
    outcomes = pd.read_csv(outcomes_path, dtype=str)
    _require_columns(predictions, PREDICTION_COLUMNS, "predictions")
    _require_columns(outcomes, OUTCOME_COLUMNS, "outcomes")

    predictions = predictions[predictions["model_version"].astype(str) == model_version].copy()
    if predictions.empty:
        raise OutcomeContractError(f"no predictions found for model version {model_version!r}")
    if predictions["transaction_id"].duplicated().any():
        raise OutcomeContractError("predictions contain duplicate transaction_id values")
    if outcomes["transaction_id"].duplicated().any():
        raise OutcomeContractError("outcomes contain duplicate transaction_id values")

    for column in ("predicted_label",):
        if not predictions[column].isin(["0", "1"]).all():
            raise OutcomeContractError(f"predictions.{column} must contain only 0 or 1")
    if not outcomes["outcome_label"].isin(["0", "1"]).all():
        raise OutcomeContractError("outcomes.outcome_label must contain only 0 or 1")
    if outcomes["observed_at"].astype(str).str.strip().eq("").any():
        raise OutcomeContractError("outcomes.observed_at must not be empty")

    predictions["predicted_label"] = predictions["predicted_label"].astype(int)
    outcomes["outcome_label"] = outcomes["outcome_label"].astype(int)
    predictions["risk_score"] = pd.to_numeric(predictions["risk_score"], errors="coerce")
    if predictions["risk_score"].isna().any():
        raise OutcomeContractError("predictions.risk_score must be numeric")

    joined = predictions.merge(outcomes, on="transaction_id", how="inner", validate="one_to_one")
    if joined.empty:
        raise OutcomeContractError("no delayed outcomes matched predictions")

    predicted = joined["predicted_label"]
    actual = joined["outcome_label"]
    tp = int(((predicted == 1) & (actual == 1)).sum())
    fp = int(((predicted == 1) & (actual == 0)).sum())
    fn = int(((predicted == 0) & (actual == 1)).sum())
    tn = int(((predicted == 0) & (actual == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "schema_version": "outcome-monitoring-report-v1",
        "model_version": model_version,
        "prediction_count": int(len(predictions)),
        "matched_outcome_count": int(len(joined)),
        "unmatched_prediction_count": int(len(predictions) - len(joined)),
        "metrics": {
            "precision": precision,
            "recall": recall,
            "coverage": float(len(joined) / len(predictions)),
        },
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "label_boundary": "outcomes are delayed monitoring labels and are not serving features",
    }


def write_report(report: dict[str, Any], output_path: str | Path) -> None:
    Path(output_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
