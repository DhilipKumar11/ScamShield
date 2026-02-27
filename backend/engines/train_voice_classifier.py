"""
ScamShield Voice Classifier Trainer
====================================
Trains a Random Forest classifier on labeled audio datasets to classify
real human vs AI-generated / voice-converted audio.

USAGE
-----
  python train_voice_classifier.py \
      --real_dir  /path/to/real_audio \
      --fake_dir  /path/to/fake_audio \
      --output    voice_model.pkl

RECOMMENDED DATASETS (free / open)
-----------------------------------
  1. Fake-or-Real (FoR):
       https://bil.eecs.yorku.ca/datasets/
  2. ASVspoof 2019 / 2021 LA subset:
       https://www.asvspoof.org/
  3. WaveFake:
       https://github.com/RUB-SysSec/WaveFake
  4. ADD (Audio Deepfake Detection):
       https://addchallenge.cn/add2022
  5. In-the-Wild Deepfake Audio:
       https://deepfake-total.com/

Place real samples in --real_dir and fake/synthetic in --fake_dir.
All common audio formats (wav, mp3, flac, ogg) are supported.
"""

import os
import argparse
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV

from feature_extractor import extract_features


AUDIO_EXTS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}


# ─── Data loading ────────────────────────────────────────────────────────────

def collect_files(directory: str) -> list[str]:
    """Return all audio file paths under a directory (recursive)."""
    paths = []
    for root, _, files in os.walk(directory):
        for f in files:
            if os.path.splitext(f)[1].lower() in AUDIO_EXTS:
                paths.append(os.path.join(root, f))
    return paths


def load_dataset(real_dir: str, fake_dir: str,
                 max_per_class: int = 2000) -> tuple:
    """
    Extract features from real and fake audio directories.
    Returns X (n_samples, 78), y (n_samples,).

    Anti-overfitting note:
      max_per_class caps each class so one class can't dominate.
    """
    print(f"\n{'='*60}")
    print("Collecting file paths …")

    real_files = collect_files(real_dir)[:max_per_class]
    fake_files = collect_files(fake_dir)[:max_per_class]

    print(f"  Real files : {len(real_files)}")
    print(f"  Fake files : {len(fake_files)}")

    X, y = [], []
    for files, label, name in [
        (real_files, 0, "REAL"),
        (fake_files, 1, "FAKE"),
    ]:
        failed = 0
        for i, fp in enumerate(files):
            vec = extract_features(fp)
            if vec is not None:
                X.append(vec)
                y.append(label)
            else:
                failed += 1
            if (i + 1) % 50 == 0:
                print(f"  [{name}] {i+1}/{len(files)} processed, {failed} failed")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"\nDataset shape : {X.shape}  |  label balance: "
          f"{(y==0).sum()} real / {(y==1).sum()} fake")
    return X, y


# ─── Model building ──────────────────────────────────────────────────────────

def build_pipeline(model_type: str = "rf") -> Pipeline:
    """
    Returns a sklearn Pipeline (scaler → calibrated classifier).

    model_type options:
      'rf'  – Random Forest  (recommended, fast, interpretable)
      'gb'  – Gradient Boosting
      'svm' – Support Vector Machine (slower, good for small data)
    """
    if model_type == "rf":
        clf = RandomForestClassifier(
            n_estimators=400,
            max_depth=18,           # limit depth → reduces overfitting
            min_samples_leaf=4,     # each leaf needs ≥4 samples
            max_features="sqrt",    # standard RF practice
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "gb":
        clf = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,          # stochastic GB → reduces overfitting
            random_state=42,
        )
    elif model_type == "svm":
        clf = SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Calibrate probabilities (Platt scaling) so output is a real probability
    calibrated = CalibratedClassifierCV(clf, method="sigmoid", cv=5)

    return Pipeline([
        ("scaler", StandardScaler()),   # z-score normalize all 78 features
        ("clf",    calibrated),
    ])


# ─── Training and evaluation ─────────────────────────────────────────────────

def train(X: np.ndarray, y: np.ndarray,
          model_type: str = "rf", test_size: float = 0.2) -> tuple:
    """
    Split data, train, evaluate, and return (pipeline, metadata_dict).

    Anti-overfitting measures applied here:
      • Stratified split (preserves class ratio in test set)
      • 5-fold CV ROC-AUC reported alongside hold-out metrics
      • StandardScaler fit ONLY on train set → no data leakage
    """
    print(f"\n{'='*60}")
    print(f"Training  model_type={model_type} …")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    print(f"  Train: {len(X_tr)}   Test: {len(X_te)}")

    pipe = build_pipeline(model_type)
    pipe.fit(X_tr, y_tr)

    # ── Evaluation ──────────────────────────────────────────────────────────
    y_pred  = pipe.predict(X_te)
    y_prob  = pipe.predict_proba(X_te)[:, 1]

    auc = roc_auc_score(y_te, y_prob)

    # 5-fold CV on the full dataset for an unbiased estimate
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(pipe, X, y, scoring="roc_auc", cv=cv, n_jobs=-1)

    print("\n── Hold-out set ──────────────────────────────────────────────")
    print(classification_report(y_te, y_pred, target_names=["Real", "Fake"]))
    print(f"Hold-out ROC-AUC : {auc:.4f}")
    print(f"Confusion matrix :\n{confusion_matrix(y_te, y_pred)}")
    print(f"\n── 5-fold CV ROC-AUC : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

    # ── Feature importance (RF only) ────────────────────────────────────────
    try:
        # Unwrap the calibrated classifier to get the base estimator
        base_rf = pipe.named_steps["clf"].calibrated_classifiers_[0].estimator
        if hasattr(base_rf, "feature_importances_"):
            importances = base_rf.feature_importances_
            top_idx = np.argsort(importances)[::-1][:10]
            FEAT_NAMES = (
                [f"MFCC_{i}_mean"  for i in range(13)] +
                [f"MFCC_{i}_std"   for i in range(13)] +
                [f"LFCC_{i}_mean"  for i in range(13)] +
                [f"LFCC_{i}_std"   for i in range(13)] +
                ["pitch_mean","pitch_std","pitch_cv","pitch_slope"] +
                ["HNR"] +
                [f"SC_band{i}_mean" for i in range(7)] +
                [f"SC_band{i}_std"  for i in range(7)] +
                ["flatness_mean","flatness_std"] +
                ["ZCR_mean","ZCR_std"] +
                ["pause_rate","pause_dur_cv","voiced_ratio"]
            )
            print("\n── Top-10 most important features ───────────────────────────")
            for rank, idx in enumerate(top_idx, 1):
                print(f"  {rank:2d}. {FEAT_NAMES[idx]:30s}  {importances[idx]:.4f}")
    except Exception:
        pass

    meta = {
        "model_type":   model_type,
        "n_features":   X.shape[1],
        "train_samples":len(X_tr),
        "test_auc":     round(auc, 4),
        "cv_auc_mean":  round(float(cv_auc.mean()), 4),
        "cv_auc_std":   round(float(cv_auc.std()),  4),
    }
    return pipe, meta


# ─── Save / load ─────────────────────────────────────────────────────────────

def save_model(pipe: Pipeline, meta: dict, output_path: str) -> None:
    payload = {"pipeline": pipe, "meta": meta}
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"\n✅ Model saved → {output_path}")
    print(f"   Metadata: {meta}")


def load_model(model_path: str) -> tuple:
    """Returns (pipeline, meta_dict)."""
    with open(model_path, "rb") as f:
        payload = pickle.load(f)
    return payload["pipeline"], payload["meta"]


# ─── CLI entry point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train ScamShield voice authenticity classifier"
    )
    parser.add_argument("--real_dir",  required=True,
                        help="Directory of REAL human voice audio files")
    parser.add_argument("--fake_dir",  required=True,
                        help="Directory of FAKE/AI-generated audio files")
    parser.add_argument("--output",    default="voice_model.pkl",
                        help="Output .pkl model file (default: voice_model.pkl)")
    parser.add_argument("--model",     default="rf",
                        choices=["rf", "gb", "svm"],
                        help="Classifier type: rf (default), gb, svm")
    parser.add_argument("--max_files", type=int, default=2000,
                        help="Max files per class (default: 2000)")
    args = parser.parse_args()

    X, y = load_dataset(args.real_dir, args.fake_dir,
                        max_per_class=args.max_files)

    if len(X) < 10:
        print("ERROR: Too few samples extracted. Check your audio directories.")
        return

    pipe, meta = train(X, y, model_type=args.model)
    save_model(pipe, meta, args.output)


if __name__ == "__main__":
    main()
