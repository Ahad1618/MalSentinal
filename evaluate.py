"""
src/evaluate.py
---------------
Evaluation module: generates confusion matrix, ROC curve, and F1/AUC metrics
on the test split. Saves plots to the plots/ directory.
"""

import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, f1_score,
    accuracy_score, roc_auc_score, classification_report
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)


def plot_confusion_matrix(y_true, y_pred, save_path: Path = None) -> plt.Figure:
    """
    Generate and optionally save a seaborn confusion matrix heatmap.
    Returns the matplotlib Figure.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Benign", "Malicious"],
        yticklabels=["Benign", "Malicious"],
        ax=ax
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"Confusion matrix saved → {save_path}")
    return fig


def plot_roc_curve(y_true, y_prob, save_path: Path = None) -> plt.Figure:
    """
    Generate and optionally save a ROC curve plot.
    Returns the matplotlib Figure.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2563EB", lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="#9CA3AF", linestyle="--", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"ROC curve saved → {save_path}")
    return fig


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Return a dict of evaluation metrics."""
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "f1":        float(f1_score(y_true, y_pred)),
        "auc":       float(roc_auc_score(y_true, y_prob)),
        "report":    classification_report(y_true, y_pred, target_names=["Benign", "Malicious"]),
    }


if __name__ == "__main__":
    from train import load_model
    from pathlib import Path

    data_dir = Path("data")
    test = np.load(data_dir / "test.npz")
    X_test, y_test = test["X"], test["y"]

    payload = load_model()
    model = payload["model"]

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = compute_metrics(y_test, y_pred, y_prob)
    log.info(f"AUC: {metrics['auc']:.4f}  F1: {metrics['f1']:.4f}")
    log.info("\n" + metrics["report"])

    plot_confusion_matrix(y_test, y_pred, PLOTS_DIR / "confusion_matrix.png")
    plot_roc_curve(y_test, y_prob, PLOTS_DIR / "roc_curve.png")
    log.info("Evaluation complete.")
