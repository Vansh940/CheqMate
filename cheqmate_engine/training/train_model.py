"""
training/train_model.py
-----------------------
Trains a 3-model soft-voting ensemble for AI text detection.

Architecture
────────────
  Model 1 — Word TF-IDF (1–2 grams) + Logistic Regression
  Model 2 — Char TF-IDF (3–5 grams) + Logistic Regression
  Model 3 — 24 Linguistic Features  + Random Forest

Final probability = weighted average (word:2 / char:2 / ling:1)

NOTE: AIEnsemble is imported from ensemble.py (not defined here)
      so joblib can deserialize it correctly when loaded in ai_detector.py.

Usage
─────
  python training/train_model.py

Requirements
────────────
  pip install scikit-learn pandas numpy joblib scipy
  Dataset:  dataset/AI_Human.csv  (columns: text, generated)
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    classification_report, confusion_matrix
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_extractor import extract_features, LinguisticTransformer
from ensemble import AIEnsemble  # ← moved here so pickle path is stable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("ModelTrainer")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "AI_Human.csv")
MODEL_DIR    = os.path.join(BASE_DIR, "model")
MODEL_PATH   = os.path.join(MODEL_DIR, "ai_detector_model.pkl")

MAX_SAMPLES = 100_000   # set None to use the full ~487k dataset


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(path: str, max_samples: int = None):
    logger.info(f"Loading dataset: {path}")
    df = pd.read_csv(path)

    logger.info(f"  Raw shape : {df.shape}")
    df = df.dropna(subset=["text", "generated"])
    df["text"]      = df["text"].astype(str)
    df["generated"] = df["generated"].astype(int)

    logger.info(f"  Class split:\n{df['generated'].value_counts().to_string()}")

    if max_samples and len(df) > max_samples:
        per_class = max_samples // 2
        frames = [
            df[df["generated"] == label].sample(
                min(int((df["generated"] == label).sum()), per_class),
                random_state=42
            )
            for label in sorted(df["generated"].unique())
        ]
        df = pd.concat(frames).reset_index(drop=True)
        logger.info(f"  Sampled to {len(df)} rows ({per_class} per class)")

    return df["text"].tolist(), df["generated"].tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def print_report(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    cm  = confusion_matrix(y_true, y_pred)

    sep = "=" * 55
    logger.info(f"\n{sep}")
    logger.info(f"  Test Accuracy : {acc * 100:.2f}%")
    logger.info(f"  ROC-AUC Score : {auc:.4f}")
    logger.info(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    logger.info(f"             Human    AI")
    logger.info(f"  Human   {cm[0][0]:>7}  {cm[0][1]:>5}")
    logger.info(f"  AI      {cm[1][0]:>7}  {cm[1][1]:>5}")
    logger.info(f"\n  Classification Report:")
    logger.info("\n" + classification_report(y_true, y_pred, target_names=["Human", "AI"]))
    logger.info(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # 1. Load
    X, y = load_dataset(DATASET_PATH, max_samples=MAX_SAMPLES)

    # 2. Split
    logger.info("Splitting 80 / 20 …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")

    # 3. Train
    logger.info("Training ensemble (3 models, sequential)…")
    model = AIEnsemble(weights=(2, 2, 1))
    model.fit(X_train, y_train)

    # 4. Evaluate
    logger.info("Evaluating on held-out test set…")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print_report(y_test, y_pred, y_prob)

    # 5. Linguistic feature importances
    try:
        imps = model.feature_importances()
        logger.info("\nTop 10 linguistic features by importance:")
        for name, imp in list(imps.items())[:10]:
            logger.info(f"  {name:<30} {imp:.4f}")
    except Exception as e:
        logger.warning(f"Could not print feature importances: {e}")

    # 6. Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)
    logger.info(f"\nModel saved → {MODEL_PATH}")


if __name__ == "__main__":
    main()