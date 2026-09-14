"""Evaluate model metrics against explicit promotion thresholds."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

try:
    from ledgerml_artifacts.store import artifact_store_from_env
except ModuleNotFoundError:
    from artifacts.ledgerml_artifacts.store import artifact_store_from_env


class EvaluationGateError(ValueError):
    """Raised when evaluation evidence is invalid or fails a configured gate."""


def _required_number(mapping: Mapping[str, Any], key: str, context: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationGateError(f"{context}.{key} must be a number")
    return float(value)


def evaluate_quality_gate(
    evaluation_path: str | Path,
    *,
    min_recall: float = 0.0,
    min_pr_auc: float = 0.0,
    max_false_negatives: int = 0,
) -> dict[str, Any]:
    """Validate evaluation evidence and enforce configured promotion gates."""
    if not 0.0 <= min_recall <= 1.0:
        raise EvaluationGateError("min_recall must be between 0 and 1")
    if not 0.0 <= min_pr_auc <= 1.0:
        raise EvaluationGateError("min_pr_auc must be between 0 and 1")
    if max_false_negatives < 0:
        raise EvaluationGateError("max_false_negatives must be >= 0")

    path = Path(evaluation_path).expanduser()
    if not path.is_file():
        raise EvaluationGateError(f"evaluation evidence does not exist: {path}")
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        raise EvaluationGateError(f"unable to read evaluation evidence {path}: {err}") from err
    if not isinstance(evidence, dict):
        raise EvaluationGateError("evaluation evidence must be a JSON object")

    metrics = evidence.get("metrics")
    confusion = evidence.get("confusion_matrix")
    if not isinstance(metrics, dict) or not isinstance(confusion, dict):
        raise EvaluationGateError("evaluation evidence must contain metrics and confusion_matrix objects")

    recall = _required_number(metrics, "recall", "metrics")
    pr_auc = _required_number(metrics, "pr_auc", "metrics")
    false_negatives = confusion.get("fn")
    if isinstance(false_negatives, bool) or not isinstance(false_negatives, int):
        raise EvaluationGateError("confusion_matrix.fn must be an integer")
    if false_negatives < 0:
        raise EvaluationGateError("confusion_matrix.fn must be >= 0")

    failures: list[str] = []
    if recall < min_recall:
        failures.append(f"recall {recall:.6f} is below minimum {min_recall:.6f}")
    if pr_auc < min_pr_auc:
        failures.append(f"pr_auc {pr_auc:.6f} is below minimum {min_pr_auc:.6f}")
    if max_false_negatives > 0 and false_negatives > max_false_negatives:
        failures.append(
            f"false negatives {false_negatives} exceed maximum {max_false_negatives}"
        )
    if failures:
        raise EvaluationGateError("; ".join(failures))

    return {
        "evaluation_path": str(path),
        "passed": True,
        "metrics": {"recall": recall, "pr_auc": pr_auc},
        "false_negatives": false_negatives,
        "thresholds": {
            "min_recall": min_recall,
            "min_pr_auc": min_pr_auc,
            "max_false_negatives": max_false_negatives,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-path",
        default=os.environ.get("LEDGERML_EVALUATION_PATH", ""),
        help="Path to evaluation-lineage.json",
    )
    parser.add_argument("--min-recall", type=float, default=float(os.environ.get("LEDGERML_EVALUATION_MIN_RECALL", "0")))
    parser.add_argument("--min-pr-auc", type=float, default=float(os.environ.get("LEDGERML_EVALUATION_MIN_PR_AUC", "0")))
    parser.add_argument("--max-false-negatives", type=int, default=int(os.environ.get("LEDGERML_EVALUATION_MAX_FALSE_NEGATIVES", "0")))
    args = parser.parse_args()

    evaluation_path = args.evaluation_path
    if os.environ.get("LEDGERML_OUTPUT_KIND") == "ObjectStore":
        temporary_file = tempfile.NamedTemporaryFile(prefix="ledgerml-evaluation-", suffix=".json", delete=False)
        temporary_path = temporary_file.name
        temporary_file.close()
        store = artifact_store_from_env()
        key = store.key(
            os.environ["LEDGERML_OUTPUT_PATH"],
            os.environ["LEDGERML_OUTPUT_ARTIFACT_VERSION"],
            "evaluation-lineage.json",
        )
        store.download_file(key, temporary_path)
        evaluation_path = temporary_path
    try:
        result = evaluate_quality_gate(
            evaluation_path,
            min_recall=args.min_recall,
            min_pr_auc=args.min_pr_auc,
            max_false_negatives=args.max_false_negatives,
        )
    except EvaluationGateError as err:
        parser.error(str(err))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
