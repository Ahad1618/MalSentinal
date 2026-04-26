"""
src/bayesian.py
---------------
Bayesian component: trains a Gaussian Naive Bayes classifier on the
import-table feature group from the EMBER feature vector.

API import features are mapped from the hash-feature region of the full
2,381-dimensional vector (indices 322–1345 cover the DLL:Function 1024-bin
hash space). Additionally, four binary sentinel features are computed from
raw PE metadata for a more interpretable Bayesian Network display.

Node variables in the Bayesian Network:
    - VirtualAlloc imported          (binary)
    - CreateRemoteThread imported    (binary)
    - WriteProcessMemory imported    (binary)
    - String entropy > 6.5           (binary)
    → Class: Malicious               (output 0/1)
"""

import logging
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple

log = logging.getLogger(__name__)

MODELS_DIR = Path("models")
BAYES_PATH = MODELS_DIR / "bayes_model.pkl"

# Indices in the 2,381-dim EMBER vector for the 1024-bin DLL:Func hash region
IMPORT_SLICE = slice(322, 1346)   # 1024 bins

# For display: interpret specific hash bins as proxy for dangerous APIs
# (Deterministic mapping via the same MD5-mod-1024 used in features.py)
import hashlib

def _api_bin(dll: str, fn: str) -> int:
    key = f"{dll.lower()}:{fn.lower()}"
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % 1024

DANGEROUS_APIS = {
    "VirtualAlloc":        _api_bin("kernel32.dll", "VirtualAlloc"),
    "CreateRemoteThread":  _api_bin("kernel32.dll", "CreateRemoteThread"),
    "WriteProcessMemory":  _api_bin("kernel32.dll", "WriteProcessMemory"),
    "OpenProcess":         _api_bin("kernel32.dll", "OpenProcess"),
    "URLDownloadToFile":   _api_bin("urlmon.dll",   "URLDownloadToFile"),
    "IsDebuggerPresent":   _api_bin("kernel32.dll", "IsDebuggerPresent"),
}

# Index in full vector for string entropy (approximate — bin 7 in string region)
STRING_ENTROPY_IDX = 2381 - 104 + 6   # str_feat[6] = entropy of all strings


def _build_bayes_features(X: np.ndarray) -> np.ndarray:
    """
    Project the full EMBER feature matrix onto the Bayesian feature subspace.

    Returns an array of shape (n_samples, n_api_features + 1) where:
      columns 0..n_api-1 = normalised import hash bins for dangerous APIs
      last column        = string entropy bin (normalised)
    """
    cols = []
    for api, bin_idx in DANGEROUS_APIS.items():
        col_idx = 322 + bin_idx  # offset into full vector
        if col_idx < X.shape[1]:
            cols.append(X[:, col_idx:col_idx+1])
        else:
            cols.append(np.zeros((X.shape[0], 1), dtype=np.float32))

    # String entropy feature
    if STRING_ENTROPY_IDX < X.shape[1]:
        cols.append(X[:, STRING_ENTROPY_IDX:STRING_ENTROPY_IDX+1])
    else:
        cols.append(np.zeros((X.shape[0], 1), dtype=np.float32))

    return np.hstack(cols).astype(np.float32)


def train_bayes(X_train: np.ndarray, y_train: np.ndarray):
    """Train GaussianNB on the Bayesian feature subspace."""
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import roc_auc_score

    X_b = _build_bayes_features(X_train)
    model = GaussianNB(var_smoothing=1e-9)
    model.fit(X_b, y_train)

    y_prob = model.predict_proba(X_b)[:, 1]
    auc = roc_auc_score(y_train, y_prob)
    log.info(f"Naive Bayes train AUC: {auc:.4f}")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "feature_names": list(DANGEROUS_APIS.keys()) + ["StringEntropy"]},
                BAYES_PATH)
    log.info(f"Bayes model saved → {BAYES_PATH}")
    return model


def load_bayes():
    """Load the saved Naive Bayes model payload."""
    if not BAYES_PATH.exists():
        raise FileNotFoundError(f"Bayes model not found. Run src/bayesian.py first.")
    return joblib.load(BAYES_PATH)


def predict_bayes(feature_vector: np.ndarray) -> Dict[str, Any]:
    """
    Run Bayesian inference on a single feature vector.

    Parameters
    ----------
    feature_vector : np.ndarray  shape (2381,)

    Returns
    -------
    dict with keys:
        malicious_prob  : float  posterior P(Malicious | API features)
        feature_values  : dict   {api_name: feature_value}
        node_probs      : dict   {api_name: P(api | Malicious)}
    """
    payload = load_bayes()
    model        = payload["model"]
    feat_names   = payload["feature_names"]

    X = feature_vector.reshape(1, -1)
    X_b = _build_bayes_features(X)
    proba = model.predict_proba(X_b)[0]
    mal_prob = float(proba[1])

    # Per-feature conditional probabilities from the model
    node_probs = {}
    for i, name in enumerate(feat_names):
        # theta_[1, i] = P(feature_i | Malicious) under GaussianNB
        if hasattr(model, "theta_"):
            node_probs[name] = float(model.theta_[1, i])

    feat_values = {name: float(X_b[0, i]) for i, name in enumerate(feat_names)}

    return {
        "malicious_prob": mal_prob,
        "feature_values": feat_values,
        "node_probs":     node_probs,
        "feature_names":  feat_names,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data_dir = Path("data")
    train_data = np.load(data_dir / "train.npz")
    train_bayes(train_data["X"], train_data["y"])
    log.info("Bayesian training complete.")
