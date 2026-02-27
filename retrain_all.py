"""
Retrain voice classifier on ALL datasets:
  - Kaggle processed (48 files: 24 normal + 24 scam)
  - Voice_Samples    (4 files:  2 Human + 2 AI)
"""
import os, sys, warnings
warnings.filterwarnings("ignore")

# Run from project root
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "backend", "engines"))

import numpy as np
from feature_extractor import extract_features
from train_voice_classifier import train, save_model

# ── Collect all data sources ──────────────────────────────────────────────────
SOURCES = {
    0: [  # Human / Real
        os.path.join(ROOT, "AI-KaggleScamAndNonscam", "processed", "normal"),
        os.path.join(ROOT, "Voice_Samples",            "processed", "Human"),
    ],
    1: [  # AI / Synthetic / Scam voice
        os.path.join(ROOT, "AI-KaggleScamAndNonscam", "processed", "scam"),
        os.path.join(ROOT, "Voice_Samples",            "processed", "AI"),
    ],
}

X, y = [], []
for label, dirs in SOURCES.items():
    lname = "HUMAN" if label == 0 else "AI"
    for d in dirs:
        if not os.path.isdir(d):
            print(f"[SKIP] {d} not found")
            continue
        files = [f for f in os.listdir(d) if f.endswith(".wav")]
        print(f"\n[{lname}] {d} ({len(files)} files)")
        for fname in files:
            fp = os.path.join(d, fname)
            vec = extract_features(fp)
            if vec is not None:
                X.append(vec)
                y.append(label)
                print(f"  OK  {fname}")
            else:
                print(f"  FAIL {fname}")

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)
print(f"\nDataset: {X.shape}  |  Human={int((y==0).sum())}  AI={int((y==1).sum())}")

if len(X) < 10:
    print("ERROR: Too few samples.")
    sys.exit(1)

# Train SVM (best for small datasets)
pipe, meta = train(X, y, model_type="svm")
out = os.path.join(ROOT, "backend", "engines", "voice_model.pkl")
save_model(pipe, meta, out)

# Also save dataset index for runtime KNN
import pickle
dataset_index = []
for label, dirs in SOURCES.items():
    lname = "AI" if label == 1 else "Human"
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if fname.endswith(".wav"):
                fp = os.path.join(d, fname)
                vec = extract_features(fp)
                if vec is not None:
                    dataset_index.append({
                        "file":  fname,
                        "label": label,        # 0=human,1=ai
                        "label_name": lname,
                        "path":  fp,
                        "vec":   vec,
                    })

index_path = os.path.join(ROOT, "backend", "engines", "dataset_index.pkl")
with open(index_path, "wb") as f:
    pickle.dump(dataset_index, f)
print(f"\nDataset index saved: {len(dataset_index)} entries -> {index_path}")
