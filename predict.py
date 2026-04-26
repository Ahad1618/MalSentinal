"""
src/predict.py
--------------
Inference engine: orchestrates the full prediction pipeline for a single PE file.

Steps:
  1. Extract 2,381 EMBER features from the binary (src/features.py)
  2. Run LightGBM classifier → malicious probability
  3. Run Naive Bayes on API features → Bayesian probability
  4. Generate SHAP waterfall explanation
  5. Return a unified result dict consumed by the Streamlit UI
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any

log = logging.getLogger(__name__)


def run_prediction(file_path: str) -> Dict[str, Any]:
    """
    Full inference pipeline on a PE binary.

    Parameters
    ----------
    file_path : str   Path to the .exe / .dll / .bin file

    Returns
    -------
    dict with keys:
        lgbm_prob       : float   LightGBM malicious probability [0,1]
        lgbm_verdict    : str     "MALICIOUS" | "BENIGN"
        bayes_prob      : float   Naive Bayes malicious probability [0,1]
        bayes_details   : dict    per-API feature values and conditional probs
        feature_vector  : ndarray shape (2381,)
        pe_info         : dict    human-readable PE metadata with text features
        malware_family  : str     Detected malware family (or "Not Available")
        shap_figure     : Figure  matplotlib waterfall chart with text feature names
        error           : str | None
    """
    result: Dict[str, Any] = {
        "lgbm_prob":     0.0,
        "lgbm_verdict":  "UNKNOWN",
        "bayes_prob":    0.0,
        "bayes_details": {},
        "feature_vector": None,
        "pe_info":       {},
        "malware_family": "Not Available (EMBER 2018 dataset)",
        "shap_figure":   None,
        "error":         None,
    }

    # ── Step 1: Feature extraction ────────────────────────────────────────────
    try:
        from features import extract_features, extract_features_dict
        feature_vector = extract_features(file_path)
        pe_info = extract_features_dict(file_path)
        result["feature_vector"] = feature_vector
        result["pe_info"] = pe_info
        log.info(f"Features extracted: shape={feature_vector.shape}")
    except Exception as exc:
        result["error"] = f"Feature extraction failed: {exc}"
        log.error(result["error"])
        return result

    # ── Step 2: LightGBM prediction ───────────────────────────────────────────
    try:
        from train import load_model
        payload   = load_model()
        lgbm_model = payload["model"]
        X = feature_vector.reshape(1, -1)
        lgbm_prob = float(lgbm_model.predict_proba(X)[0, 1])
        result["lgbm_prob"]    = lgbm_prob
        result["lgbm_verdict"] = "MALICIOUS" if lgbm_prob >= 0.5 else "BENIGN"
        result["lgbm_metrics"] = payload.get("metrics", {})
        log.info(f"LightGBM: {result['lgbm_verdict']}  p={lgbm_prob:.4f}")
    except FileNotFoundError:
        result["error"] = "LightGBM model not found. Run: python train.py"
        return result
    except Exception as exc:
        result["error"] = f"LightGBM prediction failed: {exc}"
        log.error(result["error"])
        return result

    # ── Step 3: Bayesian inference ────────────────────────────────────────────
    try:
        from bayesian import predict_bayes
        bayes_out = predict_bayes(feature_vector)
        result["bayes_prob"]    = bayes_out["malicious_prob"]
        result["bayes_details"] = bayes_out
        log.info(f"Bayes P(Malicious)={bayes_out['malicious_prob']:.4f}")
    except FileNotFoundError:
        log.warning("Bayes model not found — skipping Bayesian step.")
        result["bayes_prob"] = -1.0
    except Exception as exc:
        log.warning(f"Bayesian step failed: {exc}")
        result["bayes_prob"] = -1.0

    # ── Step 4: SHAP waterfall with text feature explanations ───────────────────
    try:
        from explain import shap_waterfall
        # Pass pe_info to get readable names for text-based features (DLLs, APIs, strings)
        fig = shap_waterfall(lgbm_model, feature_vector, pe_info=pe_info)
        result["shap_figure"] = fig
    except Exception as exc:
        log.warning(f"SHAP explanation failed: {exc}")

    # ── Step 5: Malware family detection (if available in dataset) ────────────────
    # Note: EMBER 2018 v2 dataset doesn't include family labels by default.
    # If you have an extended dataset with family info, uncomment below:
    try:
        family_model_path = Path("models") / "family_classifier.pkl"
        if family_model_path.exists():
            import joblib
            family_classifier = joblib.load(family_model_path)
            family_pred = family_classifier.predict(feature_vector.reshape(1, -1))[0]
            result["malware_family"] = str(family_pred)
            log.info(f"Malware Family: {family_pred}")
    except Exception as exc:
        log.debug(f"Malware family detection not available: {exc}")

    return result
