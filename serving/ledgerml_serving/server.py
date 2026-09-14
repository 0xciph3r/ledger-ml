"""CPU inference service for an immutable Ledger ML model artifact."""

from __future__ import annotations

import json
import math
import os
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


class InferenceError(ValueError):
    """Raised when an inference request violates the feature contract."""


class InferenceEngine:
    def __init__(
        self,
        pipeline: Any,
        feature_names: list[str],
        threshold: float,
        model_version: str,
        lineage_hash: str,
    ) -> None:
        if not feature_names:
            raise ValueError("feature_names must not be empty")
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self._pipeline = pipeline
        self._feature_names = tuple(feature_names)
        self._threshold = threshold
        self._model_version = model_version
        self._lineage_hash = lineage_hash
        self._lock = threading.Lock()
        self._requests = 0
        self._errors = 0

    @classmethod
    def from_artifact(
        cls,
        artifact_path: str | Path,
        *,
        model_version: str,
        lineage_hash: str,
    ) -> "InferenceEngine":
        import joblib

        payload = joblib.load(Path(artifact_path))
        if not isinstance(payload, dict):
            raise ValueError("model artifact must contain a dictionary payload")
        return cls(
            pipeline=payload["pipeline"],
            feature_names=list(payload["feature_names"]),
            threshold=float(payload["threshold"]),
            model_version=model_version,
            lineage_hash=lineage_hash,
        )

    @property
    def model_version(self) -> str:
        return self._model_version

    def predict(self, features: Mapping[str, Any]) -> dict[str, Any]:
        try:
            frame = self._validated_frame(features)
            probability = float(self._pipeline.predict_proba(frame)[0][1])
        except Exception:
            with self._lock:
                self._errors += 1
            raise

        with self._lock:
            self._requests += 1
        return {
            "risk_score": probability,
            "decision": "review" if probability >= self._threshold else "allow",
            "model_version": self._model_version,
            "lineage_hash": self._lineage_hash,
        }

    def _validated_frame(self, features: Mapping[str, Any]) -> pd.DataFrame:
        if not isinstance(features, Mapping):
            raise InferenceError("features must be a JSON object")
        received = set(features)
        expected = set(self._feature_names)
        missing = sorted(expected - received)
        unknown = sorted(received - expected)
        if missing:
            raise InferenceError(f"missing features: {', '.join(missing)}")
        if unknown:
            raise InferenceError(f"unknown features: {', '.join(unknown)}")
        values: dict[str, float] = {}
        for name in self._feature_names:
            value = features[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise InferenceError(f"feature {name!r} must be numeric")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise InferenceError(f"feature {name!r} must be finite")
            values[name] = numeric
        return pd.DataFrame([values], columns=list(self._feature_names))

    def metrics_text(self) -> str:
        with self._lock:
            requests = self._requests
            errors = self._errors
        return (
            "# TYPE ledgerml_inference_requests_total counter\n"
            f'ledgerml_inference_requests_total{{model_version="{self._model_version}"}} {requests}\n'
            "# TYPE ledgerml_inference_errors_total counter\n"
            f'ledgerml_inference_errors_total{{model_version="{self._model_version}"}} {errors}\n'
        )


def make_handler(engine: InferenceEngine) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _write_json(self, status: int, payload: Mapping[str, Any]) -> None:
            encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/healthz":
                self._write_json(HTTPStatus.OK, {"status": "ok"})
            elif self.path == "/readyz":
                self._write_json(HTTPStatus.OK, {"status": "ready", "model_version": engine.model_version})
            elif self.path == "/metrics":
                payload = engine.metrics_text().encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            else:
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/predict":
                self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise InferenceError("request body must be a JSON object")
                result = engine.predict(payload.get("features"))
            except (InferenceError, json.JSONDecodeError, TypeError, ValueError, AttributeError) as err:
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(err)})
                return
            self._write_json(HTTPStatus.OK, result)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def main() -> None:
    artifact_path = os.environ["LEDGERML_MODEL_PATH"]
    model_version = os.environ["LEDGERML_OUTPUT_ARTIFACT_VERSION"]
    lineage_hash = os.environ["LEDGERML_LINEAGE_HASH"]
    port = int(os.environ.get("PORT", "8080"))
    engine = InferenceEngine.from_artifact(
        artifact_path,
        model_version=model_version,
        lineage_hash=lineage_hash,
    )
    server = ThreadingHTTPServer(("0.0.0.0", port), make_handler(engine))
    server.serve_forever()


if __name__ == "__main__":
    main()
