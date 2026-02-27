"""
Retrain voice classifier with ASVspoof2019 + Kaggle + Voice_Samples
=====================================================================
Adds a balanced sample of ASVspoof2019 LA train files to the existing dataset.
Protocol format (ASVspoof2019.LA.cm.train.trn.txt):
  SPEAKER  FILE_ID  -  -  label
  label = "bonafide" (human) | "spoof" (AI-generated)

Strategy:
  - Sample ASV_SAMPLE_PER_CLASS files per class from ASVspoof train
  - Combine with ALL Kaggle + Voice_Samples (52 files)
  - Retrain SVM + rebuild dataset_index.pkl
"""
import os, sys, random, pickle, warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "backend", "engines"))

import numpy as np
from feature_extractor import extract_features
from train_voice_classifier import train, save_model

# ── Config ────────────────────────────────────────────────────────────────────
ASV_SAMPLE_PER_CLASS = 50      # how many bonafide + spoof to sample from ASVspoof
RANDOM_SEED          = 42

ASV_AUDIO_DIR = os.path.join(ROOT, "AI-ASVspoof", "LA", "ASVspoof2019_LA_train", "flac")
ASV_PROTOCOL  = os.path.join(ROOT, "AI-ASVspoof", "LA", "ASVspoof2019_LA_cm_protocols",
                              "ASVspoof2019.LA.cm.train.trn.txt")

KAGGLE_NORMAL  = os.path.join(ROOT, "AI-KaggleScamAndNonscam", "processed", "normal")
KAGGLE_SCAM    = os.path.join(ROOT, "AI-KaggleScamAndNonscam", "processed", "scam")
VS_HUMAN       = os.path.join(ROOT, "Voice_Samples", "processed", "Human")
VS_AI          = os.path.join(ROOT, "Voice_Samples", "processed", "AI")

MODEL_OUT = os.path.join(ROOT, "backend", "engines", "voice_model.pkl")
INDEX_OUT = os.path.join(ROOT, "backend", "engines", "dataset_index.pkl")

random.seed(RANDOM_SEED)

# =============================================================================
# 1. Parse ASVspoof protocol — collect file IDs per class
# =============================================================================
asv_bonafide_ids = []
asv_spoof_ids    = []

print(f"[ASVspoof] Parsing protocol: {ASV_PROTOCOL}")
with open(ASV_PROTOCOL, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        file_id = parts[1]          # e.g. LA_T_1138215
        label   = parts[4].lower()  # "bonafide" or "spoof"
        if label == "bonafide":
            asv_bonafide_ids.append(file_id)
        else:
            asv_spoof_ids.append(file_id)

print(f"[ASVspoof] Found {len(asv_bonafide_ids)} bonafide, {len(asv_spoof_ids)} spoof entries")

# Sample balanced subset
sample_bon  = random.sample(asv_bonafide_ids, min(ASV_SAMPLE_PER_CLASS, len(asv_bonafide_ids)))
sample_spoof = random.sample(asv_spoof_ids,   min(ASV_SAMPLE_PER_CLASS, len(asv_spoof_ids)))
print(f"[ASVspoof] Sampling {len(sample_bon)} bonafide + {len(sample_spoof)} spoof")


# =============================================================================
# 2. Feature extraction helper
# =============================================================================
def extract_from_dir(directory, label, ext=".wav"):
    """Extract features from all files in a directory."""
    rows = []
    if not os.path.isdir(directory):
        print(f"  [SKIP] {directory} not found")
        return rows
    files = [f for f in os.listdir(directory) if f.lower().endswith(ext)]
    lname = "Human" if label == 0 else "AI"
    print(f"\n[{lname}] {directory}  ({len(files)} files)")
    for fname in files:
        fp = os.path.join(directory, fname)
        vec = extract_features(fp)
        if vec is not None:
            rows.append({"file": fname, "label": label,
                         "label_name": lname, "path": fp, "vec": vec})
            print(f"  OK  {fname}")
        else:
            print(f"  FAIL {fname}")
    return rows


def extract_from_asv(file_ids, asv_dir, label):
    """Extract features from a list of ASVspoof file IDs (.flac files)."""
    rows = []
    lname = "Human" if label == 0 else "AI"
    ok = fail = skip = 0
    print(f"\n[ASVspoof-{lname}] Extracting {len(file_ids)} files from {asv_dir}")
    for fid in file_ids:
        fp = os.path.join(asv_dir, fid + ".flac")
        if not os.path.exists(fp):
            skip += 1
            continue
        vec = extract_features(fp)
        if vec is not None:
            rows.append({"file": fid + ".flac", "label": label,
                         "label_name": lname, "path": fp, "vec": vec})
            ok += 1
        else:
            fail += 1
    print(f"  OK={ok}  FAIL={fail}  SKIP={skip}")
    return rows


# =============================================================================
# 3. Collect ALL data
# =============================================================================
all_entries = []

# Existing Kaggle + Voice_Samples (ALL files)
all_entries += extract_from_dir(KAGGLE_NORMAL, label=0)
all_entries += extract_from_dir(KAGGLE_SCAM,   label=1)
all_entries += extract_from_dir(VS_HUMAN,       label=0)
all_entries += extract_from_dir(VS_AI,          label=1)

# ASVspoof sample
all_entries += extract_from_asv(sample_bon,   ASV_AUDIO_DIR, label=0)
all_entries += extract_from_asv(sample_spoof, ASV_AUDIO_DIR, label=1)

# =============================================================================
# 4. Build X, y arrays
# =============================================================================
X = np.array([e["vec"] for e in all_entries], dtype=np.float32)
y = np.array([e["label"] for e in all_entries], dtype=np.int32)

n_human = int((y == 0).sum())
n_ai    = int((y == 1).sum())
print(f"\n{'='*60}")
print(f"Combined dataset: {len(X)} samples  |  Human={n_human}  AI={n_ai}")
print(f"  Sources:")
print(f"    Kaggle normal:        24")
print(f"    Kaggle scam:          24")
print(f"    Voice_Samples human:   2")
print(f"    Voice_Samples AI:      2")
print(f"    ASVspoof bonafide:    {n_human - 26}")
print(f"    ASVspoof spoof:       {n_ai - 26}")
print(f"{'='*60}")

if len(X) < 20:
    print("ERROR: Too few samples to train.")
    sys.exit(1)

# =============================================================================
# 5. Train SVM
# =============================================================================
print("\nTraining SVM classifier...")
pipe, meta = train(X, y, model_type="svm")
save_model(pipe, meta, MODEL_OUT)
print(f"Model saved → {MODEL_OUT}")
print(f"Metadata: {meta}")

# =============================================================================
# 6. Save dataset index for runtime KNN
#    (KNN lookup uses ALL training entries for similarity comparison)
# =============================================================================
# Serialize vecs as plain lists to avoid numpy pickling issues
index_for_pickle = [
    {
        "file":       e["file"],
        "label":      int(e["label"]),
        "label_name": e["label_name"],
        "path":       e["path"],
        "vec":        e["vec"].tolist() if hasattr(e["vec"], "tolist") else list(e["vec"]),
    }
    for e in all_entries
]

with open(INDEX_OUT, "wb") as f:
    pickle.dump(index_for_pickle, f)

print(f"\nDataset index saved: {len(index_for_pickle)} entries → {INDEX_OUT}")
print(f"\n[DONE] Retraining complete!")
