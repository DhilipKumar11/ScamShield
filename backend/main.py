"""
ScamShield Backend — FastAPI Server
Endpoints:
  POST /analyze_audio   → Full audio analysis (upload mode)
  POST /live_chunk      → Live 5-second chunk analysis
  GET  /report/{id}     → Retrieve session report
"""
import os
import uuid
import tempfile
import time
from typing import Dict, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import logging

from engines.transcription import transcribe_audio
from engines.scam_detector import analyze_content
from engines.voice_analyzer import analyze_voice, _load_model, _load_index
from engines.risk_fusion import compute_risk, LiveRiskStabilizer

logger = logging.getLogger("scamshield.main")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

app = FastAPI(
    title="ScamShield API",
    description="AI-powered voice scam detection and synthetic voice analysis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store
sessions: Dict[str, dict] = {}


def get_time_stamp():
    return time.strftime("%H:%M:%S")


def save_temp_audio(upload_file: UploadFile) -> str:
    """Save uploaded audio to a temp file and return the path."""
    suffix = os.path.splitext(upload_file.filename or "audio.webm")[1] or ".webm"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    content = upload_file.file.read()
    tmp.write(content)
    tmp.flush()
    tmp.close()
    return tmp.name

@app.on_event("startup")
async def startup_warm_models():
    """Pre-load voice model + KNN index ONCE at startup — prevents first-request latency."""
    logger.info("[Startup] Pre-loading voice model and dataset index...")
    ml_ok  = _load_model()
    idx_ok = _load_index()
    logger.info(f"[Startup] voice_model.pkl loaded={ml_ok}  dataset_index.pkl loaded={idx_ok}")


@app.get("/")
async def root():
    return {"status": "active", "service": "ScamShield API", "version": "1.0.0"}


@app.post("/analyze_audio")
async def analyze_audio_endpoint(file: UploadFile = File(...)):
    """Full audio analysis — upload mode."""
    session_id = str(uuid.uuid4())
    
    # Save to temp file
    tmp_path = save_temp_audio(file)
    
    try:
        # 1. Transcribe
        transcription = transcribe_audio(tmp_path)
        text = transcription["text"]
        segments = transcription["segments"]
        
        # 2. Content analysis
        content_result = analyze_content(text)
        
        # 3. Voice analysis
        voice_result = analyze_voice(tmp_path)
        
        # 4. Risk fusion
        risk = compute_risk(
            content_result["score"],
            voice_result["synthetic_probability"],
            transcript=text,
            voice_verdict=voice_result.get("verdict", ""),
        )
        
        # Build transcript entries from segments
        transcript_entries = []
        for seg in segments:
            start_time = time.strftime("%H:%M:%S", time.gmtime(seg["start"]))
            seg_content = analyze_content(seg["text"])
            flagged = seg_content["score"] > 20
            transcript_entries.append({
                "time": start_time,
                "speaker": "CALLER",
                "text": seg["text"],
                "flagged": flagged,
                "categories": seg_content["categories"] if flagged else [],
            })
        
        # Build insight
        insight = content_result["insight"]
        verdict = voice_result.get("verdict", "UNCERTAIN")
        vc      = voice_result.get("verdict_confidence", 0)
        if verdict == "AI_VOICE":
            insight += f" Voice analysis confirms AI-generated speech (confidence {vc:.0f}%)."
        elif verdict == "HUMAN_VOICE":
            insight += f" Voice analysis confirms natural human speech (confidence {vc:.0f}%)."
        else:
            insight += " Voice analysis is inconclusive. Review acoustic details below."

        result = {
            "session_id":         session_id,
            "transcript":         text,
            "risk_score":         risk["final_score"],
            "content_score":      content_result["score"],
            "voice_score":        voice_result["synthetic_probability"],
            "confidence":         risk["confidence"],
            "categories":         content_result["categories"],
            "matched_keywords":   content_result["matched_keywords"],
            "threat_category":    content_result["threat_category"],
            "insight":            insight,
            # ── Ensemble verdict ──────────────────────────────────────────
            "verdict":            verdict,
            "verdict_confidence": vc,
            "ml_score":           voice_result.get("ml_score"),
            "knn_score":          voice_result.get("knn_score"),
            "nearest_matches":    voice_result.get("nearest_matches", []),
            "acoustic_details":   voice_result.get("acoustic_details", {}),
            # ── Voice panel data ──────────────────────────────────────────
            "voice_data": {
                "synthetic_probability": voice_result["synthetic_probability"],
                "pitch_variance":        voice_result["pitch_variance"],
                "mfcc_stability":        voice_result["mfcc_stability"],
                "spectral_smoothness":   voice_result["spectral_smoothness"],
                "naturalness_score":     round(100 - voice_result["synthetic_probability"], 1),
                "verdict":               verdict,
                "verdict_confidence":    vc,
                "nearest_matches":       voice_result.get("nearest_matches", []),
                "acoustic_details":      voice_result.get("acoustic_details", {}),
            },
            "transcript_entries": transcript_entries,
            "language": transcription["language"],
        }
        
        # Store session
        sessions[session_id] = result
        
        return result
    
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass

class TextAnalysisRequest(BaseModel):
    content: str

@app.post("/analyze_text")
async def analyze_text_endpoint(request: TextAnalysisRequest):
    """Analyze provided text or URL content directly."""
    text = request.content
    
    # Content analysis
    content_result = analyze_content(text)
    
    return {
        "riskScore": content_result["score"],
        "type": content_result["threat_category"] if content_result["threat_category"] != "None" else "Safe Text",
        "confidence": 90, # default high confidence for direct text
        "reasons": content_result.get("insight", "").split(". ")
    }

@app.post("/live_chunk")
async def live_chunk_endpoint(
    audio: UploadFile = File(...),    # frontend sends FormData key "audio"
    session_id: str = Form(...),
):
    """Process a 5-second live audio chunk — stable pipeline matching upload mode."""
    logger.info(f"[LiveChunk] chunk received  session={session_id}  filename={audio.filename}  size≈{audio.size}")
    tmp_path = save_temp_audio(audio)
    
    try:
        # Transcribe chunk
        transcription = transcribe_audio(tmp_path)
        text = transcription["text"]
        
        # Content analysis
        content_result = analyze_content(text)
        
        # Voice analysis
        voice_result = analyze_voice(tmp_path)
        
        # Get or create session — each session gets its own stabilizer
        if session_id not in sessions:
            sessions[session_id] = {
                "session_id":       session_id,
                "stabilizer":       LiveRiskStabilizer(window=3, require_consecutive=2),
                "all_categories":   [],
                "all_keywords":     [],
                "transcript_entries": [],
            }

        session = sessions[session_id]

        # Accumulate categories / keywords across chunks
        for cat in content_result["categories"]:
            if cat not in session["all_categories"]:
                session["all_categories"].append(cat)
        for kw in content_result["matched_keywords"]:
            if kw not in session["all_keywords"]:
                session["all_keywords"].append(kw)

        # Stabilized risk fusion (additive, voice-never-suppresses)
        risk = session["stabilizer"].update(
            content_result["score"],
            voice_result["synthetic_probability"],
            transcript=text,
            voice_verdict=voice_result.get("verdict", ""),
        )
        n = risk["chunk_number"]
        
        # ── Debug logging ─────────────────────────────────────────────────
        import numpy as np
        try:
            from engines.feature_extractor import extract_features as _ef
            _vec = _ef(tmp_path)
            if _vec is not None:
                logger.info(
                    f"[LiveChunk] session={session_id} "  
                    f"transcript={repr(text[:80])} "
                    f"raw_voice={voice_result['synthetic_probability']:.1f} "
                    f"content={content_result['score']:.1f} "
                    f"feat_mean={float(np.mean(_vec)):.4f} "
                    f"feat_std={float(np.std(_vec)):.4f} "
                    f"rolling={risk['final_score']:.1f} "
                    f"level={risk['level']}"
                )
        except Exception as _e:
            logger.debug(f"[LiveChunk] debug log skipped: {_e}")
        flagged = content_result["score"] > 20
        transcript_entry = {
            "time": get_time_stamp(),
            "speaker": "CALLER",
            "text": text,
            "flagged": flagged,
            "categories": content_result["categories"] if flagged else [],
        }
        session["transcript_entries"].append(transcript_entry)
        
        # Build insight
        insight = content_result["insight"]
        voice_verdict = voice_result.get("verdict", "")
        if voice_verdict == "AI_VOICE":
            insight += " Voice analysis indicates AI-generated speech."
        
        # Threat category
        threat_category = content_result["threat_category"]
        if risk["final_score"] >= 60 and not threat_category or threat_category == "None":
            threat_category = "Financial Vishing (Voice Phishing)"
        
        return {
            "transcript_entry": transcript_entry,
            "risk_score":       risk["final_score"],
            "content_score":    round(content_result["score"], 1),
            "voice_score":      round(voice_result["synthetic_probability"], 1),
            "confidence":       risk["confidence"],
            "categories":       session["all_categories"],
            "insight":          insight,
            "threat_category":  threat_category,
            "voice_data": {
                "synthetic_probability": voice_result["synthetic_probability"],
                "pitch_variance":        voice_result["pitch_variance"],
                "mfcc_stability":        voice_result["mfcc_stability"],
                "spectral_smoothness":   voice_result["spectral_smoothness"],
            },
            "chunk_number": n,
        }
    
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass


@app.get("/report/{session_id}")
async def get_report(session_id: str):
    """Retrieve the full report for a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
