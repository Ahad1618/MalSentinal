"""
src/explain.py
--------------
SHAP explainability module.

Provides:
  - shap_waterfall()        : per-prediction waterfall chart
  - shap_feature_importance(): top-N global feature importance bar chart
"""

import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


def _load_feature_names(n: int = 2381, pe_info: dict = None) -> list:
    """
    Load feature names from data/feature_names.txt, or generate generic names.
    If pe_info is provided, enhance feature names with readable text features
    (DLLs, APIs, strings, URLs, registry keys, file paths).
    """
    path = Path("data") / "feature_names.txt"
    names = []
    
    if path.exists():
        names = path.read_text().splitlines()
        if len(names) >= n:
            return names[:n]
    
    # Generate generic names for EMBER features
    names = [f"feature_{i}" for i in range(n)]
    
    # Enhance with readable text features if pe_info available
    if pe_info:
        # Enhance import-related features (indices vary but we map a few)
        dll_names = pe_info.get("dll_names", [])
        if dll_names:
            # DLL imports are typically in feature range 256-512 (hashed)
            for i, dll in enumerate(dll_names[:10]):  # Show top 10 DLLs
                idx = 256 + min(i * 25, 200)  # Spread across import region
                if idx < len(names):
                    names[idx] = f"Import: {dll}"
        
        # Enhance suspicious API features
        suspicious_apis = pe_info.get("suspicious_apis", [])
        if suspicious_apis:
            for i, api in enumerate(suspicious_apis[:10]):  # Show top 10 APIs
                idx = 512 + min(i * 15, 150)  # Different region
                if idx < len(names):
                    names[idx] = f"Suspicious API: {api}"
        
        # Enhance string features
        extracted_strings = pe_info.get("extracted_strings", [])
        indicators = pe_info.get("suspicious_indicators", [])
        
        if indicators:
            for i, (ind_type, ind_value) in enumerate(indicators[:5]):
                idx = 2300 + min(i * 15, 70)  # String feature region (near end)
                if idx < len(names):
                    display_val = ind_value[:30] + "..." if len(ind_value) > 30 else ind_value
                    names[idx] = f"{ind_type}: {display_val}"
    
    return names


def shap_waterfall(
    model,
    feature_vector: np.ndarray,
    feature_names: Optional[list] = None,
    pe_info: Optional[dict] = None,
    max_display: int = 20,
) -> plt.Figure:
    """
    Compute SHAP values for a single prediction and return a waterfall Figure.
    Enhanced to show readable text feature names (DLLs, APIs, strings, URLs, etc.)

    Parameters
    ----------
    model          : trained LightGBM model (LGBMClassifier)
    feature_vector : np.ndarray  shape (2381,)
    feature_names  : optional list of feature name strings
    pe_info        : optional dict with extracted text features for readable names
    max_display    : number of top features to display

    Returns
    -------
    matplotlib Figure with text-feature-aware SHAP waterfall chart
    """
    try:
        import shap
    except ImportError:
        log.warning("shap not installed. Returning placeholder figure.")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Install shap to see explanations", ha="center", va="center")
        return fig

    if feature_names is None:
        # Use pe_info to enhance feature names with text features
        feature_names = _load_feature_names(len(feature_vector), pe_info=pe_info)

    explainer = shap.TreeExplainer(model)
    X = feature_vector.reshape(1, -1)
    shap_values = explainer.shap_values(X)

    # Handle NaN values in SHAP values
    if isinstance(shap_values, list):
        shap_values = [np.nan_to_num(arr, nan=0.0) for arr in shap_values]
    else:
        shap_values = np.nan_to_num(shap_values, nan=0.0)

    # LightGBM binary: shap_values may be list [benign, malicious] or single array
    if isinstance(shap_values, list):
        sv = shap_values[1][0]   # malicious class
    elif shap_values.ndim == 3:
        sv = shap_values[0, :, 1]  # for models returning (n_samples, n_features, n_classes)
    else:
        sv = shap_values[0]

    # Sort by absolute SHAP value
    idx = np.argsort(np.abs(sv))[::-1][:max_display]
    top_sv = sv[idx]
    top_names = [feature_names[i] if i < len(feature_names) else f"f{i}" for i in idx]

    fig, ax = plt.subplots(figsize=(9, max_display * 0.4 + 1.5))
    colors = ["#DC2626" if v > 0 else "#2563EB" for v in top_sv]
    y_pos = np.arange(len(top_sv))

    ax.barh(y_pos, top_sv[::-1], color=colors[::-1], height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value  (red: malicious,  blue: benign)", fontsize=10)
    ax.set_title("Prediction Explanation - Top Feature Contributions (with Text Features)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def shap_feature_importance(model, X_sample: np.ndarray, feature_names: Optional[list] = None,
                             top_n: int = 20) -> plt.Figure:
    """
    Compute mean |SHAP| values over a sample and return a bar chart Figure.

    Parameters
    ----------
    model         : trained LightGBM model
    X_sample      : np.ndarray  shape (n_samples, 2381)  — subsample for speed
    feature_names : optional list of feature name strings
    top_n         : number of top features to show

    Returns
    -------
    matplotlib Figure
    """
    try:
        import shap
    except ImportError:
        log.warning("shap not installed.")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Install shap to see feature importance", ha="center", va="center")
        return fig

    if feature_names is None:
        feature_names = _load_feature_names(X_sample.shape[1])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    mean_abs = np.abs(sv).mean(axis=0)
    idx = np.argsort(mean_abs)[::-1][:top_n]
    top_names = [feature_names[i] if i < len(feature_names) else f"f{i}" for i in idx]
    top_vals  = mean_abs[idx]

    fig, ax = plt.subplots(figsize=(9, top_n * 0.4 + 1.5))
    y_pos = np.arange(top_n)
    ax.barh(y_pos, top_vals[::-1], color="#6366F1", height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP value|", fontsize=10)
    ax.set_title(f"Top {top_n} Global Feature Importances", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig
