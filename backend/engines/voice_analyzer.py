"""
Voice Authenticity Engine — ScamShield Backend
================================================
Ensemble: SVM model (50%) + KNN against all 52 dataset files (50%)
Gives DEFINITIVE verdict: AI_VOICE / HUMAN_VOICE / UNCERTAIN

Output dict:
  {
    "verdict":           "AI_VOICE" | "HUMAN_VOICE" | "UNCERTAIN"
    "verdict_confidence": 0-100
    "synthetic_probability": 0-100
    "method":            "ensemble" | "ml" | "heuristic"
    "model_auc":         float
    "nearest_matches":   list of top-7 nearest dataset files with labels
    "acoustic_details":  dict of pitch/jitter/shimmer etc for UI panels
    "pitch_variance":    0-100
    "mfcc_stability":    0-100
    "spectral_smoothness": 0-100
    "details":           dict
  }
"""
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa


def _sanitize(obj):
    """Recursively convert numpy scalars/arrays to native Python types for JSON."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# ── Paths ─────────────────────────────────────────────────────────────────────
_DIR        = os.path.dirname(__file__)
_MODEL_PATH = os.path.join(_DIR, "voice_model.pkl")
_INDEX_PATH = os.path.join(_DIR, "dataset_index.pkl")

# ── Lazy-loaded singletons ────────────────────────────────────────────────────
_pipeline      = None
_model_meta    = {}
_dataset_index = None   # list of {file, label, label_name, vec}
_index_matrix  = None   # (N, 78) numpy array for fast distance
_index_labels  = None   # (N,) numpy int array


def _load_model():
    global _pipeline, _model_meta
    if _pipeline is not None:
        return True
    if not os.path.exists(_MODEL_PATH):
        return False
    try:
        with open(_MODEL_PATH, "rb") as f:
            payload = pickle.load(f)
        _pipeline   = payload["pipeline"]
        _model_meta = payload.get("meta", {})
        print(f"[VoiceAnalyzer] ML model loaded (AUC={_model_meta.get('cv_auc_mean','?')})")
        return True
    except Exception as e:
        print(f"[VoiceAnalyzer] Could not load ML model: {e}")
        return False


def _load_index():
    global _dataset_index, _index_matrix, _index_labels
    if _dataset_index is not None:
        return True
    if not os.path.exists(_INDEX_PATH):
        print("[VoiceAnalyzer] No dataset_index.pkl — KNN disabled")
        return False
    try:
        with open(_INDEX_PATH, "rb") as f:
            _dataset_index = pickle.load(f)
        vecs = np.array([e["vec"] for e in _dataset_index], dtype=np.float32)
        # L2-normalise for cosine similarity
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
        _index_matrix = vecs / norms
        _index_labels = np.array([e["label"] for e in _dataset_index], dtype=np.int32)
        print(f"[VoiceAnalyzer] Dataset index loaded: {len(_dataset_index)} samples")
        return True
    except Exception as e:
        print(f"[VoiceAnalyzer] Could not load dataset index: {e}")
        return False


# ── KNN comparison against full dataset ───────────────────────────────────────

def _knn_compare(vec: np.ndarray, k: int = 7) -> tuple[float, list[dict]]:
    """
    Compare feature vector against ALL dataset samples.
    Returns (knn_ai_prob_0_100, nearest_matches_list)
    nearest_matches sorted by similarity (highest first).
    """
    if not _load_index() or _index_matrix is None:
        return 50.0, []

    # Normalise query vector
    q = vec.astype(np.float32)
    q_norm = q / (np.linalg.norm(q) + 1e-10)

    # Cosine similarity to all N dataset vectors (dot product on normalised)
    similarities = _index_matrix @ q_norm   # (N,)
    distances    = 1.0 - similarities        # 0=identical, 2=opposite

    top_k_idx = np.argsort(distances)[:k]

    matches = []
    for rank, idx in enumerate(top_k_idx, 1):
        entry = _dataset_index[int(idx)]
        matches.append({
            "rank":       rank,
            "file":       entry["file"],
            "label":      entry["label_name"],   # "AI" or "Human"
            "similarity": round(float(similarities[idx]) * 100, 1),
            "distance":   round(float(distances[idx]), 4),
        })

    # Weighted KNN vote (closer = more weight)
    ai_votes = 0.0
    total_w  = 0.0
    for i, idx in enumerate(top_k_idx):
        w = 1.0 / (distances[idx] + 1e-6)
        if _index_labels[idx] == 1:   # AI
            ai_votes += w
        total_w += w

    knn_ai_prob = (ai_votes / total_w * 100) if total_w > 0 else 50.0
    return knn_ai_prob, matches


# ── ML inference ──────────────────────────────────────────────────────────────

def _ml_predict(vec: np.ndarray) -> float | None:
    """Returns 0-100 probability of AI voice, or None on failure."""
    if not _load_model():
        return None
    try:
        prob = float(_pipeline.predict_proba(vec.reshape(1, -1))[0, 1]) * 100
        return round(min(100.0, max(0.0, prob)), 1)
    except Exception as e:
        print(f"[VoiceAnalyzer] ML inference error: {e}")
        return None


# ── Acoustic detail extraction (for UI panels) ────────────────────────────────

def _acoustic_details(y: np.ndarray, sr: int) -> dict:
    """Extract human-readable acoustic features for front-end panels."""
    hop = 256
    try:
        # Pitch
        f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr, frame_length=2048)
        f0c = f0[~np.isnan(f0)] if f0 is not None and len(f0) > 0 else np.array([150.])
        p_mean = float(np.mean(f0c))
        p_std  = float(np.std(f0c))
        p_cv   = p_std / p_mean if p_mean > 0 else 0.0    # low → AI

        # Jitter (cycle-to-cycle pitch variation)
        jitter = float(np.mean(np.abs(np.diff(f0c))) / p_mean) if p_mean > 0 and len(f0c) > 1 else 0.0

        # Shimmer (amplitude variation)
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        shimmer = float(np.std(rms) / (np.mean(rms) + 1e-10))

        # HNR approximation
        clip = y[:sr]
        ac   = np.correlate(clip, clip, mode='full')
        ac   = ac[len(ac)//2:]
        if ac[0] > 1e-12 and np.max(ac[1:]) > 0:
            peak = np.max(ac[1:])
            hnr  = float(10 * np.log10(peak / (ac[0] - peak + 1e-12)))
        else:
            hnr = 0.0

        # Spectral
        sf_ = librosa.feature.spectral_flatness(y=y, hop_length=hop)[0]
        sc  = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
        sc_cv  = float(np.std(sc) / (np.mean(sc) + 1e-10))

        # MFCC delta
        mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)
        md     = librosa.feature.delta(mfcc)
        mfcc_d_std = float(np.mean(np.std(md, axis=1)))

        # ACF periodicity
        frames = librosa.util.frame(y, frame_length=2048, hop_length=512, axis=0)[:30]
        acf_peaks = []
        for frame in frames:
            ac2 = np.correlate(frame, frame, mode='full')
            ac2 = ac2[len(ac2)//2:]; ac2 /= (ac2[0] + 1e-10)
            acf_peaks.append(float(np.max(ac2[1:])))
        acf_strength = float(np.mean(acf_peaks)) if acf_peaks else 0.5

        voiced_ratio = float(np.sum(~np.isnan(f0)) / len(f0)) if f0 is not None and len(f0) > 0 else 0.5

        return {
            "pitch_mean_hz":        round(p_mean, 1),
            "pitch_std_hz":         round(p_std, 2),
            "pitch_cv_pct":         round(p_cv * 100, 2),      # <8 AI, >15 Human
            "jitter_pct":           round(jitter * 100, 4),     # <0.3 AI, >0.8 Human
            "shimmer_pct":          round(shimmer * 100, 2),    # <3 AI, >5 Human
            "hnr_db":               round(hnr, 2),              # >20 AI (too clean)
            "mfcc_delta_std":       round(mfcc_d_std, 4),      # <3 AI, >5 Human
            "spectral_centroid_cv": round(sc_cv * 100, 2),
            "spectral_flatness":    round(float(np.mean(sf_)), 6),
            "acf_periodicity":      round(acf_strength, 4),    # >0.72 AI
            "voiced_ratio_pct":     round(voiced_ratio * 100, 1),
            "duration_s":           round(len(y) / sr, 2),
        }
    except Exception as e:
        print(f"[VoiceAnalyzer] Acoustic detail error: {e}")
        return {}


def _acoustic_to_panel_scores(d: dict) -> dict:
    """Convert raw acoustic features to 0-100 UI panel scores."""
    # pitch_variance: high = human, low = AI → panel shows humanness
    p_cv = d.get("pitch_cv_pct", 10)
    pitch_variance = min(100, p_cv * 4)          # 25% CV → 100

    # mfcc_stability: high = AI (too smooth)
    mfcc_d = d.get("mfcc_delta_std", 5)
    mfcc_stability = min(100, max(0, 100 - mfcc_d * 8))

    # spectral_smoothness: high = AI
    sc_cv  = d.get("spectral_centroid_cv", 10)
    spectral_smoothness = min(100, max(0, 100 - sc_cv * 2))

    return {
        "pitch_variance":      round(pitch_variance, 1),
        "mfcc_stability":      round(mfcc_stability, 1),
        "spectral_smoothness": round(spectral_smoothness, 1),
    }


# ── Verdict logic ─────────────────────────────────────────────────────────────

def _make_verdict(ensemble_score: float) -> tuple[str, float]:
    """
    ensemble_score: 0-100 probability of AI voice.
    Returns (verdict_string, confidence_pct)
    Thresholds tuned to avoid ambiguous middle-ground labels.
    """
    conf = abs(ensemble_score - 50) * 2   # 0-100, 100 = maximally confident
    if ensemble_score >= 58:
        return "AI_VOICE", round(min(conf, 100), 1)
    elif ensemble_score <= 42:
        return "HUMAN_VOICE", round(min(conf, 100), 1)
    else:
        return "UNCERTAIN", round(min(conf, 100), 1)


# ── Public API ────────────────────────────────────────────────────────────────

def analyze_voice(file_path: str) -> dict:
    """
    Full ensemble analysis.
    Compares against ALL dataset files + SVM model.
    Returns definitive verdict: AI_VOICE / HUMAN_VOICE / UNCERTAIN
    """
    # 1. Extract 78-dim feature vector
    try:
        from engines.feature_extractor import extract_features
        vec = extract_features(file_path)
    except Exception:
        try:
            from feature_extractor import extract_features as _ef
            vec = _ef(file_path)
        except Exception as e:
            print(f"[VoiceAnalyzer] Feature extraction failed: {e}")
            return _default_result(f"Feature extraction failed: {e}")

    if vec is None:
        return _default_result("Feature extraction returned None")

    # 2. ML model prediction
    ml_prob  = _ml_predict(vec)

    # 3. Full KNN against all dataset files
    knn_prob, matches = _knn_compare(vec, k=7)

    # 4. Ensemble (weighted average)
    if ml_prob is not None:
        ensemble = ml_prob * 0.55 + knn_prob * 0.45
        method   = "ensemble"
    else:
        ensemble = knn_prob
        method   = "knn_only"

    ensemble = round(min(100, max(0, ensemble)), 1)

    # 5. Verdict
    verdict, confidence = _make_verdict(ensemble)

    # 6. Acoustic details for UI panels
    try:
        y, sr = librosa.load(file_path, sr=16000, mono=True, duration=60)
        acoustic = _acoustic_details(y, sr)
        panel    = _acoustic_to_panel_scores(acoustic)
    except Exception as e:
        print(f"[VoiceAnalyzer] Librosa load error: {e}")
        acoustic = {}
        panel    = {"pitch_variance": 50, "mfcc_stability": 50, "spectral_smoothness": 50}

    return _sanitize({
        "verdict":               verdict,
        "verdict_confidence":    confidence,
        "synthetic_probability": ensemble,
        "ml_score":              ml_prob,
        "knn_score":             round(knn_prob, 1),
        "method":                method,
        "model_auc":             _model_meta.get("cv_auc_mean", "N/A"),
        "nearest_matches":       matches,
        "acoustic_details":      acoustic,
        "pitch_variance":        panel["pitch_variance"],
        "mfcc_stability":        panel["mfcc_stability"],
        "spectral_smoothness":   panel["spectral_smoothness"],
        "details":               {
            "n_dataset_files": len(_dataset_index) if _dataset_index else 0,
            "feature_dims":    78,
            "method":          method,
        },
    })


def _default_result(reason: str = "") -> dict:
    return {
        "verdict":               "UNCERTAIN",
        "verdict_confidence":    0.0,
        "synthetic_probability": 50.0,
        "ml_score":              None,
        "knn_score":             50.0,
        "method":                "failed",
        "model_auc":             "N/A",
        "nearest_matches":       [],
        "acoustic_details":      {},
        "pitch_variance":        50.0,
        "mfcc_stability":        50.0,
        "spectral_smoothness":   50.0,
        "details":               {"error": reason},
    }
