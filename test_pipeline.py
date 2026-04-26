"""
tests/test_pipeline.py
----------------------
pytest unit tests for the MalSentinel pipeline.
Tests feature extraction shape, model loading, prediction output,
Bayesian output structure, and SHAP output shape.
"""

import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

TARGET_DIM = 2381


# ── Feature extraction ────────────────────────────────────────────────────────

class TestFeatureExtraction:
    def test_feature_vector_shape_on_missing_file(self):
        """extract_features must return zero vector of correct dim for bad paths."""
        from features import extract_features
        vec = extract_features("/nonexistent/path/file.exe")
        assert vec.shape == (TARGET_DIM,), f"Expected ({TARGET_DIM},), got {vec.shape}"

    def test_feature_vector_dtype(self):
        from features import extract_features
        vec = extract_features("/nonexistent/path/file.exe")
        assert vec.dtype == np.float32

    def test_no_nan_or_inf(self):
        from features import extract_features
        vec = extract_features("/nonexistent/path/file.exe")
        assert not np.any(np.isnan(vec)), "Feature vector contains NaN"
        assert not np.any(np.isinf(vec)), "Feature vector contains Inf"

    def test_extract_features_dict_returns_dict(self):
        from features import extract_features_dict
        info = extract_features_dict("/nonexistent/path/file.exe")
        assert isinstance(info, dict)
        for key in ("file_size_bytes", "num_sections", "imports", "exports",
                    "suspicious_apis", "machine_type"):
            assert key in info, f"Missing key: {key}"


# ── Bayesian feature projection ───────────────────────────────────────────────

class TestBayesianFeatures:
    def test_build_bayes_features_shape(self):
        from bayesian import _build_bayes_features, DANGEROUS_APIS
        X = np.zeros((5, TARGET_DIM), dtype=np.float32)
        X_b = _build_bayes_features(X)
        # Should have len(DANGEROUS_APIS) + 1 columns
        expected_cols = len(DANGEROUS_APIS) + 1
        assert X_b.shape == (5, expected_cols), (
            f"Expected (5, {expected_cols}), got {X_b.shape}"
        )

    def test_build_bayes_features_no_nan(self):
        from bayesian import _build_bayes_features
        X = np.random.rand(10, TARGET_DIM).astype(np.float32)
        X_b = _build_bayes_features(X)
        assert not np.any(np.isnan(X_b))


# ── Model loading (skipped if models not trained) ─────────────────────────────

@pytest.mark.skipif(
    not Path("models/rf_model.pkl").exists(),
    reason="Model not trained yet"
)
class TestLGBMModel:
    def test_model_loads(self):
        from train import load_model
        payload = load_model()
        assert "model" in payload
        assert "metrics" in payload

    def test_model_predicts_shape(self):
        from train import load_model
        payload = load_model()
        model = payload["model"]
        X_dummy = np.random.rand(3, TARGET_DIM).astype(np.float32)
        proba = model.predict_proba(X_dummy)
        assert proba.shape == (3, 2), f"Expected (3, 2), got {proba.shape}"

    def test_probabilities_sum_to_one(self):
        from train import load_model
        payload = load_model()
        model = payload["model"]
        X_dummy = np.random.rand(10, TARGET_DIM).astype(np.float32)
        proba = model.predict_proba(X_dummy)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(10), atol=1e-5)


@pytest.mark.skipif(
    not Path("models/bayes_model.pkl").exists(),
    reason="Bayes model not trained yet"
)
class TestBayesModel:
    def test_bayes_loads(self):
        from bayesian import load_bayes
        payload = load_bayes()
        assert "model" in payload
        assert "feature_names" in payload

    def test_bayes_predict_output_keys(self):
        from bayesian import predict_bayes
        vec = np.random.rand(TARGET_DIM).astype(np.float32)
        out = predict_bayes(vec)
        for key in ("malicious_prob", "feature_values", "node_probs", "feature_names"):
            assert key in out, f"Missing key: {key}"

    def test_bayes_probability_in_range(self):
        from bayesian import predict_bayes
        vec = np.random.rand(TARGET_DIM).astype(np.float32)
        out = predict_bayes(vec)
        assert 0.0 <= out["malicious_prob"] <= 1.0


# ── Report generation ─────────────────────────────────────────────────────────

class TestReportGeneration:
    def _dummy_result(self):
        return {
            "lgbm_prob":     0.75,
            "lgbm_verdict":  "MALICIOUS",
            "bayes_prob":    0.68,
            "pe_info":       {"file_size_bytes": 12345, "num_sections": 4,
                              "machine_type": "x86", "entry_point": "0x1000",
                              "has_tls": False, "has_debug": False, "has_resources": True,
                              "suspicious_apis": ["VirtualAlloc", "CreateRemoteThread"]},
            "lgbm_metrics":  {"accuracy": 0.98, "f1": 0.97, "auc": 0.99},
            "bayes_details": {"feature_values": {"VirtualAlloc": 1.0}, "node_probs": {},
                              "feature_names": ["VirtualAlloc"]},
            "shap_figure":   None,
            "error":         None,
        }

    def test_report_returns_bytes(self):
        try:
            from report.generate import generate_report
        except ImportError:
            pytest.skip("fpdf2 not installed")
        pdf_bytes = generate_report(self._dummy_result())
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 1000, "PDF too small — likely empty"

    def test_report_is_valid_pdf(self):
        try:
            from report.generate import generate_report
        except ImportError:
            pytest.skip("fpdf2 not installed")
        pdf_bytes = generate_report(self._dummy_result())
        assert pdf_bytes[:4] == b"%PDF", "Output is not a valid PDF"
