"""Synthetic fraud trainer for Ledger ML.

This module implements a deterministic, local-only fraud model training slice for the
teaching milestones. It uses synthetic transaction data and a small logistic
regression pipeline to produce:

1) an immutable model artifact (`model.joblib`)
2) an evaluation + lineage JSON record (`evaluation-lineage.json`)

Why not accuracy-only:
Fraud detection is intentionally imbalanced, so a model can show high accuracy while
missing most fraud. We therefore report precision, recall, F1, PR-AUC, ROC-AUC, and
confusion matrix counts.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES: Sequence[str] = (
    "amount",
    "hour",
    "merchant_risk",
    "distance_km",
    "tx_count_1h",
    "tx_count_24h",
    "device_risk",
)

REQUIRED_ENV_VARS: Sequence[str] = (
    "LEDGERML_TASK",
    "LEDGERML_DATASET_KIND",
    "LEDGERML_DATASET_NAME",
    "LEDGERML_DATASET_PATH",
    "LEDGERML_DATASET_VERSION",
    "LEDGERML_OUTPUT_KIND",
    "LEDGERML_OUTPUT_NAME",
    "LEDGERML_OUTPUT_PATH",
    "LEDGERML_OUTPUT_ARTIFACT_VERSION",
    "LEDGERML_TRAINING_IMAGE_DIGEST",
    "LEDGERML_CONFIGURATION_DIGEST",
)

VERSION_TOKEN_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class ConfigurationError(ValueError):
    """Raised when required LEDGERML_* inputs are missing or invalid."""


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    task: str
    dataset_kind: str
    dataset_name: str
    dataset_path: str
    dataset_version: str
    output_kind: str
    output_name: str
    output_path: str
    output_artifact_version: str
    training_image_digest: str
    configuration_digest_input: str
    random_seed: int
    num_samples: int
    test_size: float
    threshold: float

    @property
    def artifact_root(self) -> Path:
        """Directory where output artifacts are written."""
        base = Path(self.output_name).expanduser()
        if not base.is_absolute():
            base = Path.cwd() / base
        relative_path = self.output_path.strip("/")
        if relative_path:
            base = base / relative_path
        return base / self.output_artifact_version


def _require_env(env: Mapping[str, str], key: str) -> str:
    value = env.get(key, "").strip()
    if not value:
        raise ConfigurationError(f"Missing required environment variable: {key}")
    return value


def _parse_positive_int(raw: str, name: str) -> int:
    try:
        value = int(raw)
    except ValueError as err:
        raise ConfigurationError(f"{name} must be an integer, got {raw!r}") from err
    if value <= 0:
        raise ConfigurationError(f"{name} must be > 0, got {value}")
    return value


def _parse_ratio(raw: str, name: str) -> float:
    try:
        value = float(raw)
    except ValueError as err:
        raise ConfigurationError(f"{name} must be a float, got {raw!r}") from err
    if not (0.0 < value < 1.0):
        raise ConfigurationError(f"{name} must be between 0 and 1 (exclusive), got {value}")
    return value


def load_training_config(env: Mapping[str, str]) -> TrainingConfig:
    """Load and validate trainer configuration from LEDGERML_* environment."""
    for required_key in REQUIRED_ENV_VARS:
        _require_env(env, required_key)

    task = _require_env(env, "LEDGERML_TASK")
    if task != "fraud-scoring":
        raise ConfigurationError(
            f"Unsupported LEDGERML_TASK={task!r}; expected 'fraud-scoring'"
        )

    output_artifact_version = _require_env(env, "LEDGERML_OUTPUT_ARTIFACT_VERSION")
    if not VERSION_TOKEN_RE.fullmatch(output_artifact_version):
        raise ConfigurationError(
            "LEDGERML_OUTPUT_ARTIFACT_VERSION must match [A-Za-z0-9._-]+ for immutable artifact naming"
        )

    config = TrainingConfig(
        task=task,
        dataset_kind=_require_env(env, "LEDGERML_DATASET_KIND"),
        dataset_name=_require_env(env, "LEDGERML_DATASET_NAME"),
        dataset_path=_require_env(env, "LEDGERML_DATASET_PATH"),
        dataset_version=_require_env(env, "LEDGERML_DATASET_VERSION"),
        output_kind=_require_env(env, "LEDGERML_OUTPUT_KIND"),
        output_name=_require_env(env, "LEDGERML_OUTPUT_NAME"),
        output_path=_require_env(env, "LEDGERML_OUTPUT_PATH"),
        output_artifact_version=output_artifact_version,
        training_image_digest=_require_env(env, "LEDGERML_TRAINING_IMAGE_DIGEST"),
        configuration_digest_input=_require_env(env, "LEDGERML_CONFIGURATION_DIGEST"),
        random_seed=_parse_positive_int(env.get("LEDGERML_RANDOM_SEED", "42"), "LEDGERML_RANDOM_SEED"),
        num_samples=_parse_positive_int(env.get("LEDGERML_NUM_SAMPLES", "15000"), "LEDGERML_NUM_SAMPLES"),
        test_size=_parse_ratio(env.get("LEDGERML_TEST_SIZE", "0.2"), "LEDGERML_TEST_SIZE"),
        threshold=_parse_ratio(env.get("LEDGERML_CLASSIFICATION_THRESHOLD", "0.15"), "LEDGERML_CLASSIFICATION_THRESHOLD"),
    )
    return config


def generate_synthetic_transactions(config: TrainingConfig) -> tuple[pd.DataFrame, np.ndarray, dict[str, str]]:
    """Generate deterministic synthetic transaction data.

    Assumptions:
    - Fraud is rare and driven by high-risk merchants, unusual device risk,
      travel distance, and short-window transaction velocity.
    - Night-time transactions have slightly elevated fraud risk.
    - No real customer or cardholder records are used.
    """
    rng = np.random.default_rng(config.random_seed)
    n = config.num_samples

    amount = rng.lognormal(mean=3.2, sigma=1.0, size=n)
    hour = rng.integers(0, 24, size=n)
    merchant_risk = rng.beta(2.0, 5.0, size=n)
    distance_km = rng.gamma(shape=2.0, scale=18.0, size=n)
    tx_count_1h = rng.poisson(lam=0.6 + merchant_risk * 2.3, size=n)
    tx_count_24h = tx_count_1h + rng.poisson(lam=1.5 + merchant_risk * 3.0, size=n)
    device_risk = rng.beta(1.5, 4.5, size=n)

    is_night = ((hour < 6) | (hour >= 23)).astype(float)

    logit = (
        -7.1
        + 2.00 * merchant_risk
        + 0.0060 * amount
        + 0.020 * distance_km
        + 0.45 * tx_count_1h
        + 0.15 * tx_count_24h
        + 2.20 * device_risk
        + 0.80 * is_night
    )
    fraud_probability = 1.0 / (1.0 + np.exp(-logit))
    labels = rng.binomial(1, np.clip(fraud_probability, 1e-4, 0.99))

    features = pd.DataFrame(
        {
            "amount": amount.astype(float),
            "hour": hour.astype(float),
            "merchant_risk": merchant_risk.astype(float),
            "distance_km": distance_km.astype(float),
            "tx_count_1h": tx_count_1h.astype(float),
            "tx_count_24h": tx_count_24h.astype(float),
            "device_risk": device_risk.astype(float),
        }
    )

    assumptions = {
        "label_definition": "fraud = Bernoulli(sigmoid(weighted risk score over synthetic features))",
        "imbalance_design": "intercept and coefficients chosen so fraud prevalence remains low but non-trivial",
        "privacy": "synthetic-only data; no customer PII or real transactions",
    }
    return features, labels.astype(int), assumptions


def _build_pipeline(config: TrainingConfig) -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=600,
                    solver="liblinear",
                    random_state=config.random_seed,
                ),
            ),
        ]
    )


def _computed_configuration_digest(config: TrainingConfig) -> str:
    payload = {
        "task": config.task,
        "dataset": {
            "kind": config.dataset_kind,
            "name": config.dataset_name,
            "path": config.dataset_path,
            "version": config.dataset_version,
        },
        "output": {
            "kind": config.output_kind,
            "name": config.output_name,
            "path": config.output_path,
            "artifact_version": config.output_artifact_version,
        },
        "trainer": {
            "algorithm": "logistic_regression",
            "random_seed": config.random_seed,
            "num_samples": config.num_samples,
            "test_size": config.test_size,
            "threshold": config.threshold,
        },
        "training_image_digest": config.training_image_digest,
    }
    normalized = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(normalized).hexdigest()


def run_training(config: TrainingConfig) -> dict[str, Path]:
    """Train and emit immutable artifact + machine-readable evaluation output."""
    config.artifact_root.mkdir(parents=True, exist_ok=True)

    features, labels, assumptions = generate_synthetic_transactions(config)
    fraud_rate = float(labels.mean())
    if fraud_rate <= 0.0 or fraud_rate >= 0.5:
        raise RuntimeError(
            f"Synthetic class balance is out of expected range for fraud detection: fraud_rate={fraud_rate:.4f}"
        )

    train_features, test_features, train_labels, test_labels = train_test_split(
        features,
        labels,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=labels,
    )

    pipeline = _build_pipeline(config)
    pipeline.fit(train_features, train_labels)

    probabilities = pipeline.predict_proba(test_features)[:, 1]
    predictions = (probabilities >= config.threshold).astype(int)

    if len(np.unique(test_labels)) < 2:
        raise RuntimeError(
            "Test split contains a single class; cannot compute ROC-AUC/PR-AUC reliably."
        )

    cm = confusion_matrix(test_labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))

    model_path = config.artifact_root / "model.joblib"
    evaluation_path = config.artifact_root / "evaluation-lineage.json"

    model_payload = {
        "pipeline": pipeline,
        "feature_names": list(FEATURE_NAMES),
        "threshold": config.threshold,
        "dataset_version": config.dataset_version,
        "output_artifact_version": config.output_artifact_version,
    }
    joblib.dump(model_payload, model_path)

    computed_digest = _computed_configuration_digest(config)
    evaluation = {
        "task": config.task,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "kind": config.dataset_kind,
            "name": config.dataset_name,
            "path": config.dataset_path,
            "version": config.dataset_version,
            "generator": "ledgerml-synthetic-fraud-v1",
            "assumptions": assumptions,
        },
        "lineage": {
            "training_image_digest": config.training_image_digest,
            "configuration_digest_input": config.configuration_digest_input,
            "configuration_digest_computed": computed_digest,
            "configuration_digest_matches": config.configuration_digest_input == computed_digest,
            "output": {
                "kind": config.output_kind,
                "name": config.output_name,
                "path": config.output_path,
                "artifact_version": config.output_artifact_version,
            },
            "model_artifact_path": str(model_path),
            "feature_names": list(FEATURE_NAMES),
        },
        "class_balance": {
            "total_samples": int(config.num_samples),
            "fraud_count": int(labels.sum()),
            "non_fraud_count": int((labels == 0).sum()),
            "fraud_rate": fraud_rate,
        },
        "threshold": config.threshold,
        "metrics": {
            "precision": float(precision_score(test_labels, predictions, zero_division=0)),
            "recall": float(recall_score(test_labels, predictions, zero_division=0)),
            "f1": float(f1_score(test_labels, predictions, zero_division=0)),
            "pr_auc": float(average_precision_score(test_labels, probabilities)),
            "roc_auc": float(roc_auc_score(test_labels, probabilities)),
        },
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "train_test_split": {
            "train_rows": int(train_features.shape[0]),
            "test_rows": int(test_features.shape[0]),
            "random_seed": config.random_seed,
            "test_size": config.test_size,
        },
        "model": {
            "algorithm": "logistic_regression",
            "preprocessing": ["standard_scaler"],
            "sklearn_version": sklearn.__version__,
        },
    }

    with evaluation_path.open("w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2, sort_keys=True)
        f.write("\n")

    return {
        "model_artifact_path": model_path,
        "evaluation_path": evaluation_path,
    }


def run_cli(env: Mapping[str, str] | None = None) -> int:
    """CLI entrypoint used by the training container and local runs."""
    active_env = env if env is not None else os.environ
    try:
        config = load_training_config(active_env)
    except ConfigurationError as err:
        print(f"Configuration error: {err}", file=sys.stderr)
        return 2

    outputs = run_training(config)
    print(f"Training completed. Model artifact: {outputs['model_artifact_path']}")
    print(f"Evaluation record: {outputs['evaluation_path']}")
    return 0
