"""
Voice_Samples: Standardize + Deep Acoustic Analysis
Converts all files to 16kHz mono WAV and computes detailed feature comparison
between AI-generated and Human voices.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import librosa
import soundfile as sf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, "processed")
os.makedirs(OUT_DIR, exist_ok=True)

FILES = {
    "AI": [
        os.path.join(SCRIPT_DIR, "AI_Voices", "AiGeneratedVoice.mpeg"),
        os.path.join(SCRIPT_DIR, "AI_Voices",  "Ai_gen_Voice.mp3"),
    ],
    "Human": [
        os.path.join(SCRIPT_DIR, "Human_Voices", "HumanVoice1.ogg"),
        os.path.join(SCRIPT_DIR, "Human_Voices", "humanVoice.ogg"),
    ],
}

SR = 16000

def standardize(src, dst):
    y, _ = librosa.load(src, sr=SR, mono=True)
    sf.write(dst, y, SR, subtype="PCM_16")
    return y

def deep_features(y, sr):
    """Compute 20 detailed acoustic features used to distinguish AI vs Human."""
    hop = 256

    # 1. Pitch (pyin)
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr, frame_length=2048)
    f0c = f0[~np.isnan(f0)] if f0 is not None and len(f0)>0 else np.array([0.])
    pitch_mean  = float(np.mean(f0c))
    pitch_std   = float(np.std(f0c))
    pitch_cv    = pitch_std / pitch_mean if pitch_mean > 0 else 0.0

    # 2. Jitter (cycle-to-cycle pitch variation)
    jitter = float(np.mean(np.abs(np.diff(f0c))) / pitch_mean) if pitch_mean > 0 and len(f0c)>1 else 0.0

    # 3. Shimmer (amplitude variation)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    shimmer = float(np.std(rms) / np.mean(rms)) if np.mean(rms) > 0 else 0.0

    # 4. HNR (Harmonic-to-Noise Ratio)
    acf = np.correlate(y[:sr], y[:sr], mode='full')
    acf = acf[len(acf)//2:]
    hnr = float(10 * np.log10(np.max(acf[1:]) / (acf[0] - np.max(acf[1:]) + 1e-10))) if acf[0] > 1e-10 else 0.0

    # 5. MFCC stats
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_stability = float(np.mean(np.std(mfcc_delta, axis=1)))

    # 6. Spectral features
    sc  = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    sb  = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop)[0]
    sf_ = librosa.feature.spectral_flatness(y=y, hop_length=hop)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop)[0]

    sc_cv  = float(np.std(sc) / np.mean(sc)) if np.mean(sc)>0 else 0
    sf_cv  = float(np.std(sf_) / np.mean(sf_)) if np.mean(sf_)>0 else 0
    sf_mean= float(np.mean(sf_))

    # 7. ZCR
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop)[0]
    zcr_mean = float(np.mean(zcr))
    zcr_std  = float(np.std(zcr))

    # 8. Tempo & rhythm
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(tempo) if hasattr(tempo, '__float__') else float(tempo[0]) if len(tempo) > 0 else 0.0

    # 9. Voiced ratio
    voiced_ratio = float(np.sum(~np.isnan(f0)) / len(f0)) if f0 is not None and len(f0)>0 else 0.5

    # 10. ACF peak strength (periodicity)
    hop_acf = 512
    frames = librosa.util.frame(y, frame_length=2048, hop_length=hop_acf, axis=0)
    acf_peaks = []
    for frame in frames[:50]:
        ac = np.correlate(frame, frame, mode='full')
        ac = ac[len(ac)//2:]
        if ac[0] > 1e-10:
            ac /= ac[0]
            acf_peaks.append(float(np.max(ac[1:])))
    acf_mean = float(np.mean(acf_peaks)) if acf_peaks else 0.5

    return {
        "pitch_mean_hz":      round(pitch_mean, 1),
        "pitch_std_hz":       round(pitch_std, 2),
        "pitch_cv_%":         round(pitch_cv * 100, 2),
        "jitter_%":           round(jitter * 100, 4),
        "shimmer_%":          round(shimmer * 100, 2),
        "hnr_db":             round(hnr, 2),
        "mfcc_delta_std":     round(mfcc_stability, 4),
        "spectral_centroid_cv": round(sc_cv * 100, 2),
        "spectral_flatness_mean": round(sf_mean, 6),
        "spectral_flatness_cv": round(sf_cv, 3),
        "zcr_mean":           round(zcr_mean, 5),
        "zcr_std":            round(zcr_std, 5),
        "tempo_bpm":          round(tempo, 1),
        "voiced_ratio_%":     round(voiced_ratio * 100, 1),
        "acf_periodicity":    round(acf_mean, 4),
        "duration_s":         round(len(y)/sr, 2),
    }

def verdict(f):
    """Simple rule-based verdict from features for display."""
    ai_score = 0
    reasons = []
    # Pitch CV: AI voices have very low variability
    if f["pitch_cv_%"] < 8:
        ai_score += 2; reasons.append(f"Low pitch variance ({f['pitch_cv_%']}% < 8%)")
    elif f["pitch_cv_%"] > 15:
        ai_score -= 1; reasons.append(f"High pitch variance ({f['pitch_cv_%']}% > 15% → human)")

    # Jitter: AI very low
    if f["jitter_%"] < 0.3:
        ai_score += 2; reasons.append(f"Near-zero jitter ({f['jitter_%']}% < 0.3%)")
    elif f["jitter_%"] > 0.8:
        ai_score -= 1; reasons.append(f"Natural jitter ({f['jitter_%']}% → human)")

    # Shimmer: AI very low
    if f["shimmer_%"] < 3:
        ai_score += 2; reasons.append(f"Low shimmer ({f['shimmer_%']}% < 3%)")
    elif f["shimmer_%"] > 7:
        ai_score -= 1; reasons.append(f"Natural shimmer ({f['shimmer_%']}% → human)")

    # HNR: AI very high (perfectly harmonic)
    if f["hnr_db"] > 20:
        ai_score += 1; reasons.append(f"Very high HNR ({f['hnr_db']} dB → synthesized)")

    # Spectral flatness stability
    if f["spectral_flatness_cv"] < 0.5:
        ai_score += 1; reasons.append(f"Stable spectral flatness (CV={f['spectral_flatness_cv']} < 0.5)")

    # MFCC stability: AI more stable transitions
    if f["mfcc_delta_std"] < 3.0:
        ai_score += 1; reasons.append(f"Very smooth MFCC transitions ({f['mfcc_delta_std']})")

    # ACF periodicity: AI more periodic
    if f["acf_periodicity"] > 0.72:
        ai_score += 1; reasons.append(f"High periodicity ACF={f['acf_periodicity']} > 0.72")
    elif f["acf_periodicity"] < 0.55:
        ai_score -= 1; reasons.append(f"Low periodicity ACF={f['acf_periodicity']} → human")

    if ai_score >= 4:
        v = "🔴 AI VOICE"
    elif ai_score <= -1:
        v = "🟢 HUMAN VOICE"
    else:
        v = "🟡 UNCERTAIN"
    return v, ai_score, reasons

# ── MAIN ─────────────────────────────────────────────────────────────────────

results = {"AI": [], "Human": []}

print("\n" + "="*70)
print("  VOICE SAMPLE DEEP ACOUSTIC ANALYSIS")
print("="*70)

for label, files in FILES.items():
    for src in files:
        name = os.path.basename(src)
        stem = os.path.splitext(name)[0]
        dst  = os.path.join(OUT_DIR, label, f"{stem}.wav")
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        print(f"\n[{label}] {name}")
        try:
            y = standardize(src, dst)
            print(f"  ✅ Converted → {os.path.relpath(dst)}")
            feats = deep_features(y, SR)
            v, score, reasons = verdict(feats)
            results[label].append({"file": name, "feats": feats, "verdict": v, "score": score})

            print(f"  VERDICT  : {v}  (score={score})")
            print(f"  Duration : {feats['duration_s']}s")
            print(f"  ─── Pitch ────────────────────────")
            print(f"    Mean       : {feats['pitch_mean_hz']} Hz")
            print(f"    Std Dev    : {feats['pitch_std_hz']} Hz")
            print(f"    CV (var%)  : {feats['pitch_cv_%']}%  ← AI<8%, Human>12%")
            print(f"    Jitter     : {feats['jitter_%']}%   ← AI<0.3%, Human>0.8%")
            print(f"  ─── Amplitude ────────────────────")
            print(f"    Shimmer    : {feats['shimmer_%']}%  ← AI<3%, Human>5%")
            print(f"    HNR        : {feats['hnr_db']} dB   ← AI>20dB (too perfect)")
            print(f"  ─── Spectral ─────────────────────")
            print(f"    Centroid CV: {feats['spectral_centroid_cv']}%")
            print(f"    Flatness   : {feats['spectral_flatness_mean']:.6f} (CV={feats['spectral_flatness_cv']})")
            print(f"    MFCC Δ std : {feats['mfcc_delta_std']}  ← AI<3, Human>5")
            print(f"  ─── Periodicity ──────────────────")
            print(f"    ACF peak   : {feats['acf_periodicity']}  ← AI>0.72, Human<0.65")
            print(f"    Voiced %   : {feats['voiced_ratio_%']}%")
            print(f"  ─── Reasons ──────────────────────")
            for r in reasons:
                print(f"    • {r}")
        except Exception as e:
            print(f"  ❌ FAILED: {e}")

# Summary comparison
print("\n" + "="*70)
print("  ACOUSTIC DIFFERENCE SUMMARY: AI vs Human")
print("="*70)
ai_feats = [r["feats"] for r in results["AI"]]
hu_feats = [r["feats"] for r in results["Human"]]

if ai_feats and hu_feats:
    keys = list(ai_feats[0].keys())
    print(f"\n{'Feature':<28} {'AI (avg)':>12} {'Human (avg)':>12}  Interpretation")
    print("-"*70)
    for k in keys:
        ai_avg = np.mean([f[k] for f in ai_feats])
        hu_avg = np.mean([f[k] for f in hu_feats])
        diff = "← AI lower" if ai_avg < hu_avg else "← AI higher"
        print(f"  {k:<26} {ai_avg:>12.4f} {hu_avg:>12.4f}  {diff}")

print("\n✅ Standardized WAVs saved to Voice_Samples/processed/")
