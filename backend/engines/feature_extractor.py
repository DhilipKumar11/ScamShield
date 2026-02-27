"""
ScamShield Voice Feature Extractor
Extracts a fixed-length feature vector from an audio file for
real-vs-synthetic classification.

Features (total 78 dimensions):
  - MFCC (13 coeffs):  mean + std           = 26
  - LFCC (13 coeffs):  mean + std           = 26
  - Pitch (F0):        mean, std, cv, slope  =  4
  - Harmonic-to-Noise Ratio (HNR)            =  1
  - Spectral Contrast (7 bands): mean + std  = 14
  - Spectral Flatness                        =  2  (mean + std)
  - ZCR                                      =  2  (mean + std)
  - Pause / silence irregularity             =  3  (pause_rate, pause_dur_cv, voiced_ratio)
  ──────────────────────────────────────────────
  Total                                      = 78
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa
from scipy.signal import lfilter


# ─── LFCC helper ─────────────────────────────────────────────────────────────

def _linear_filterbank(sr: int, n_fft: int, n_filters: int = 24) -> np.ndarray:
    """Build a linear-spaced triangular filterbank (n_filters x n_fft//2+1)."""
    low  = 0.0
    high = sr / 2.0
    freqs = np.linspace(low, high, n_filters + 2)
    bins  = np.floor((n_fft + 1) * freqs / sr).astype(int)
    fbank = np.zeros((n_filters, n_fft // 2 + 1))
    for m in range(1, n_filters + 1):
        f_m_minus = bins[m - 1]
        f_m       = bins[m]
        f_m_plus  = bins[m + 1]
        for k in range(f_m_minus, f_m):
            denom = f_m - f_m_minus
            fbank[m - 1, k] = (k - f_m_minus) / denom if denom else 0
        for k in range(f_m, f_m_plus):
            denom = f_m_plus - f_m
            fbank[m - 1, k] = (f_m_plus - k) / denom if denom else 0
    return fbank


def compute_lfcc(y: np.ndarray, sr: int, n_lfcc: int = 13, n_fft: int = 2048,
                 hop_length: int = 512, n_filters: int = 24) -> np.ndarray:
    """Compute LFCC matrix (n_lfcc x T) using a linear filterbank + DCT."""
    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
    fbank = _linear_filterbank(sr, n_fft, n_filters)
    # Clamp fbank to stft shape
    fbank = fbank[:, : stft.shape[0]]
    log_filterbank_energy = np.log(fbank @ stft + 1e-10)   # (n_filters, T)
    # Apply DCT-2
    from scipy.fftpack import dct
    lfcc = dct(log_filterbank_energy, type=2, axis=0, norm='ortho')[:n_lfcc]
    return lfcc                                              # (n_lfcc, T)


# ─── Pitch helpers ──────────────────────────────────────────────────────────

def _pitch_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Return [mean, std, cv, linear_slope] of voiced F0 frames."""
    try:
        f0, voiced, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr,
            frame_length=2048,
        )
        f0_clean = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        if len(f0_clean) < 4:
            return np.array([0., 0., 0., 0.])
        mu  = float(np.mean(f0_clean))
        std = float(np.std(f0_clean))
        cv  = std / mu if mu > 0 else 0.0
        t   = np.arange(len(f0_clean))
        slope = float(np.polyfit(t, f0_clean, 1)[0]) if len(t) > 1 else 0.0
        return np.array([mu, std, cv, slope])
    except Exception:
        return np.array([0., 0., 0., 0.])


# ─── HNR ────────────────────────────────────────────────────────────────────

def _hnr(y: np.ndarray, sr: int) -> float:
    """Estimate Harmonic-to-Noise Ratio via autocorrelation."""
    try:
        frame_length = int(0.025 * sr)
        hop          = int(0.010 * sr)
        frames = librosa.util.frame(y, frame_length=frame_length,
                                    hop_length=hop, axis=0)
        hnr_vals = []
        for frame in frames:
            acf = np.correlate(frame, frame, mode='full')
            acf = acf[len(acf) // 2:]
            if acf[0] < 1e-10:
                continue
            # Find first peak after lag 0
            lag0  = acf[0]
            peak  = np.max(acf[1:])
            if lag0 - peak > 0:
                hnr_db = 10 * np.log10(peak / (lag0 - peak + 1e-10))
            else:
                hnr_db = 0.0
            hnr_vals.append(hnr_db)
        return float(np.mean(hnr_vals)) if hnr_vals else 0.0
    except Exception:
        return 0.0


# ─── Pause / voicing irregularity ───────────────────────────────────────────

def _pause_features(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Returns [pause_rate, pause_dur_cv, voiced_ratio].
    pause_rate   = number of silence segments per second
    pause_dur_cv = coefficient of variation of silence segment durations
    voiced_ratio = fraction of time that is voiced
    """
    try:
        intervals = librosa.effects.split(y, top_db=25)         # voiced segments
        total_dur = len(y) / sr
        voiced_dur = sum((e - s) / sr for s, e in intervals)
        voiced_ratio = voiced_dur / total_dur if total_dur > 0 else 1.0

        # Silence (pause) durations
        pauses = []
        for i in range(1, len(intervals)):
            gap = (intervals[i][0] - intervals[i - 1][1]) / sr
            if gap > 0.05:  # > 50 ms counts as a pause
                pauses.append(gap)
        pause_rate   = len(pauses) / total_dur if total_dur > 0 else 0.0
        pause_dur_cv = (np.std(pauses) / np.mean(pauses)) if len(pauses) > 1 else 0.0
        return np.array([pause_rate, pause_dur_cv, voiced_ratio])
    except Exception:
        return np.array([0., 0., 1.])


# ─── Main feature extractor ─────────────────────────────────────────────────

def extract_features(file_path: str, sr_target: int = 22050) -> np.ndarray | None:
    """
    Load an audio file and return a 1-D feature vector (length 78).
    Returns None if extraction fails.
    """
    try:
        y, sr = librosa.load(file_path, sr=sr_target, mono=True, duration=60)
        if len(y) < sr * 0.5:
            return None

        hop = 512

        # 1. MFCC (13) → mean + std = 26
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)
        mfcc_feats = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])  # 26

        # 2. LFCC (13) → mean + std = 26
        lfcc = compute_lfcc(y, sr, n_lfcc=13, hop_length=hop)
        lfcc_feats = np.concatenate([lfcc.mean(axis=1), lfcc.std(axis=1)])  # 26

        # 3. Pitch features = 4
        pitch_feats = _pitch_features(y, sr)

        # 4. HNR = 1
        hnr_feat = np.array([_hnr(y, sr)])

        # 5. Spectral Contrast (7 bands × 2) = 14
        sc = librosa.feature.spectral_contrast(y=y, sr=sr,
                                               n_bands=6, hop_length=hop)
        sc_feats = np.concatenate([sc.mean(axis=1), sc.std(axis=1)])        # 14

        # 6. Spectral Flatness = 2
        sf = librosa.feature.spectral_flatness(y=y, hop_length=hop)[0]
        sf_feats = np.array([sf.mean(), sf.std()])

        # 7. ZCR = 2
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop)[0]
        zcr_feats = np.array([zcr.mean(), zcr.std()])

        # 8. Pause / voicing = 3
        pause_feats = _pause_features(y, sr)

        vec = np.concatenate([
            mfcc_feats,    # 26
            lfcc_feats,    # 26
            pitch_feats,   #  4
            hnr_feat,      #  1
            sc_feats,      # 14
            sf_feats,      #  2
            zcr_feats,     #  2
            pause_feats,   #  3
        ])                 # = 78

        # Replace NaN / Inf
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        return vec.astype(np.float32)

    except Exception as e:
        print(f"[FeatureExtractor] Error on {file_path}: {e}")
        return None
