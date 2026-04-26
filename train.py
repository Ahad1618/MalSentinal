"""
train.py
--------
Train a RandomForest classifier on the EMBER-style dataset.
Loads pre-processed train/test .npz files produced by setup_data.py,
trains the model, saves it, and logs evaluation metrics.
"""

import logging
import numpy as np
from pathlib import Path
import joblib

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATA_DIR   = Path("data")
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "rf_model.pkl"


def load_splits():
    """Load train and test numpy arrays from .npz files."""
    train = np.load(DATA_DIR / "train.npz")
    test  = np.load(DATA_DIR / "test.npz")
    return train["X"], train["y"], test["X"], test["y"]


def train(X_train, y_train, X_val, y_val):
    """
    Train a LightGBM binary classifier.

    Parameters
    ----------
    X_train, y_train : training features and labels
    X_val,   y_val   : validation features and labels

    Returns
    -------
    Trained LGBMClassifier
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        log.error("scikit-learn not installed. Run: pip install scikit-learn")
        raise

    params = {
        "n_estimators":      100,
        "max_depth":         10,
        "min_samples_split": 20,
        "min_samples_leaf":  10,
        "random_state":      42,
        "n_jobs":            -1,
    }

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    """Compute and log accuracy, F1, and AUC-ROC on the test set."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    log.info(f"Accuracy : {acc:.4f}")
    log.info(f"F1 Score : {f1:.4f}")
    log.info(f"AUC-ROC  : {auc:.4f}")
    log.info("\n" + classification_report(y_test, y_pred, target_names=["Benign", "Malicious"]))

    return {"accuracy": acc, "f1": f1, "auc": auc}


def save_model(model, metrics: dict):
    """Persist the trained model and its metrics."""
    MODELS_DIR.mkdir(exist_ok=True)
    payload = {"model": model, "metrics": metrics}
    joblib.dump(payload, MODEL_PATH)
    log.info(f"Model saved → {MODEL_PATH}")


def load_model():
    """
    Load the saved LightGBM model payload.

    Returns
    -------
    dict with keys 'model' and 'metrics'
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")
    return joblib.load(MODEL_PATH)


if __name__ == "__main__":
    log.info("Loading data …")
    X_train, y_train, X_test, y_test = load_splits()

    log.info("Training RandomForest …")
    model = train(X_train, y_train, X_test, y_test)

    log.info("Evaluating …")
    metrics = evaluate(model, X_test, y_test)

    save_model(model, metrics)
    log.info("Training complete.")
