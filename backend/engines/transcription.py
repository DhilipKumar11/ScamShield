"""
Speech-to-Text Engine using OpenAI Whisper (English only).
Audio is pre-loaded with librosa (avoids whisper's internal ffmpeg subprocess call).
Falls back to empty transcription if whisper is not available or fails.
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa

_model      = None
_whisper_ok = False

# Lazy import
try:
    import whisper as _whisper_lib
    _whisper_ok = True
except ImportError:
    _whisper_lib = None


def get_model():
    global _model
    if not _whisper_ok:
        return None
    if _model is None:
        print("[Transcription] Loading Whisper base model…")
        _model = _whisper_lib.load_model("base")
        print("[Transcription] Whisper model ready.")
    return _model


def transcribe_audio(file_path: str) -> dict:
    """
    Transcribe audio using Whisper.
    Audio is pre-loaded with librosa so whisper never calls ffmpeg directly.

    Returns: {"text": str, "language": str, "segments": list[dict]}
    """
    if not _whisper_ok:
        print("[Transcription] Whisper not installed — skipping.")
        return {"text": "", "language": "unknown", "segments": []}

    try:
        model = get_model()
        if model is None:
            return {"text": "", "language": "unknown", "segments": []}

        # Load audio with librosa → 16 kHz mono float32 numpy array
        # This avoids whisper calling ffmpeg via subprocess internally.
        audio_np, _ = librosa.load(file_path, sr=16000, mono=True, dtype=np.float32)

        # Whisper accepts a numpy array directly (float32, 16 kHz)
        result = model.transcribe(
            audio_np,
            verbose=False,
            fp16=False,      # CPU mode
            language="en",   # English only — skip auto-detection
        )

        segments = [
            {
                "start": round(seg["start"], 2),
                "end":   round(seg["end"],   2),
                "text":  seg["text"].strip(),
            }
            for seg in result.get("segments", [])
        ]

        return {
            "text":     result.get("text", "").strip(),
            "language": result.get("language", "unknown"),
            "segments": segments,
        }

    except Exception as e:
        print(f"[Transcription] Error during transcription: {e}")
        return {"text": "", "language": "unknown", "segments": []}
