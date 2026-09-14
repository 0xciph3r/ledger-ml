import unittest

from serving.ledgerml_serving.server import InferenceEngine, InferenceError


class FakePipeline:
    def predict_proba(self, frame):
        self.columns = list(frame.columns)
        return [[0.2, 0.8]]


class InferenceEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = InferenceEngine(
            pipeline=FakePipeline(),
            feature_names=["amount", "velocity"],
            threshold=0.5,
            model_version="model-v1",
            lineage_hash="sha256:test",
        )

    def test_feature_contract_is_table_driven(self) -> None:
        cases = [
            ("missing", {"amount": 1}, "missing features"),
            ("unknown", {"amount": 1, "velocity": 2, "extra": 3}, "unknown features"),
            ("boolean", {"amount": True, "velocity": 2}, "must be numeric"),
            ("infinite", {"amount": float("inf"), "velocity": 2}, "must be finite"),
        ]
        for name, features, expected in cases:
            with self.subTest(case=name):
                with self.assertRaisesRegex(InferenceError, expected):
                    self.engine.predict(features)

    def test_valid_prediction_contains_lineage_and_decision(self) -> None:
        result = self.engine.predict({"amount": 100, "velocity": 2})
        self.assertEqual(result["decision"], "review")
        self.assertEqual(result["model_version"], "model-v1")
        self.assertEqual(result["lineage_hash"], "sha256:test")
        self.assertIn("ledgerml_inference_requests_total", self.engine.metrics_text())


if __name__ == "__main__":
    unittest.main()
