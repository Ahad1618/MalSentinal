"""
setup_data.py
-------------
Data pipeline for loading the EMBER 2018 v2 dataset from a local parquet file.
Splits into train/test sets and saves them for downstream training steps.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
PARQUET_PATH = Path("test_ember_2018_v2_features.parquet")  # local dataset
TRAIN_NPZ = DATA_DIR / "train.npz"
TEST_NPZ  = DATA_DIR / "test.npz"

# EMBER 2018 v2 has 2,381 feature columns; label column is "Label"
LABEL_COL = "Label"
UNLABELED = -1          # EMBER uses -1 for unlabeled samples


def load_parquet(path: Path) -> pd.DataFrame:
    """Load EMBER parquet file and return a DataFrame."""
    log.info(f"Loading dataset from {path} …")
    if not path.exists():
        log.error(f"Dataset not found at {path}. "
                  "Place test_ember_2018_v2_features.parquet in the project root.")
        sys.exit(1)
    df = pd.read_parquet(path)
    log.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


def prepare(df: pd.DataFrame):
    """
    Filter out unlabeled rows, separate features from labels,
    split into train/test, and save .npz files.
    """
    # Drop unlabeled rows (label == -1)
    labeled = df[df[LABEL_COL] != UNLABELED].copy()
    log.info(f"Labeled samples: {len(labeled):,}  "
             f"(malicious={int(labeled[LABEL_COL].sum()):,}, "
             f"benign={int((labeled[LABEL_COL]==0).sum()):,})")

    feature_cols = [c for c in labeled.columns if c != LABEL_COL]
    X = labeled[feature_cols].values.astype(np.float32)
    y = labeled[LABEL_COL].values.astype(np.int8)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log.info(f"Train: {X_train.shape}  Test: {X_test.shape}")

    DATA_DIR.mkdir(exist_ok=True)
    np.savez(TRAIN_NPZ, X=X_train, y=y_train)
    np.savez(TEST_NPZ,  X=X_test,  y=y_test)
    log.info(f"Saved → {TRAIN_NPZ}  {TEST_NPZ}")

    # Also persist feature column names for downstream use
    feat_path = DATA_DIR / "feature_names.txt"
    feat_path.write_text("\n".join(feature_cols))
    log.info(f"Feature names saved → {feat_path}")

    return feature_cols


if __name__ == "__main__":
    df = load_parquet(PARQUET_PATH)
    prepare(df)
    log.info("Data pipeline complete.")
