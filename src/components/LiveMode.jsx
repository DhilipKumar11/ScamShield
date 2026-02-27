import React, { useState, useRef, useEffect, useCallback } from "react";
import { analyzeContent } from "../audioEngine";
import RiskMeter from "./RiskMeter";
import AlertBanner from "./AlertBanner";
import VoiceAuthPanel from "./VoiceAuthPanel";
import TranscriptDisplay from "./TranscriptDisplay";
import RiskBreakdown from "./RiskBreakdown";
import SafetyPanel from "./SafetyPanel";

const BACKEND = "http://localhost:8000";
const CHUNK_INTERVAL_MS = 5000; // 5-second chunks
const MIME_TYPE = "audio/webm;codecs=opus";

export default function LiveMode() {
  const [recording, setRecording] = useState(false);
  const [transcriptEntries, setTranscriptEntries] = useState([]);
  const [riskScore, setRiskScore] = useState(0);
  const [contentScore, setContentScore] = useState(0);
  const [voiceScore, setVoiceScore] = useState(0);
  const [confidence, setConfidence] = useState(0);
  const [categories, setCategories] = useState([]);
  const [insight, setInsight] = useState("");
  const [threatCategory, setThreatCategory] = useState("");
  const [voiceData, setVoiceData] = useState({});
  const [timer, setTimer] = useState(0);
  const [chunkCount, setChunkCount] = useState(0);
  const [micError, setMicError] = useState("");
  const [interimText, setInterimText] = useState("");
  const [backendActive, setBackendActive] = useState(false);

  // Refs (survive re-renders without triggering them)
  const streamRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const chunkIntervalRef = useRef(null);
  const timerRef = useRef(null);
  const recognitionRef = useRef(null);
  const sessionIdRef = useRef(null);
  const allTextRef = useRef("");
  const latestVoiceRef = useRef({}); // last voice_data from backend

  /* ── helpers ────────────────────────────────────────── */
  const formatTime = (s) => {
    const m = Math.floor(s / 60)
      .toString()
      .padStart(2, "0");
    const sec = (s % 60).toString().padStart(2, "0");
    return `${m}:${sec}`;
  };

  const getTimeStamp = () =>
    new Date().toLocaleTimeString("en-US", {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });

  /* ── send one recorded blob to backend ─────────────── */
  const sendChunk = useCallback(async (blob) => {
    if (!blob || blob.size < 500) return; // skip near-empty blobs

    const formData = new FormData();
    formData.append("audio", blob, "chunk.webm");
    formData.append("session_id", sessionIdRef.current);

    try {
      const resp = await fetch(`${BACKEND}/live_chunk`, {
        method: "POST",
        body: formData,
      });
      if (!resp.ok) {
        console.warn("[LiveMode] /live_chunk HTTP", resp.status);
        return;
      }

      const data = await resp.json();
      console.log("[LiveMode] chunk response:", data);

      // ── Update UI from rolling backend score ──────────
      const rollingScore = data.risk_score ?? 0; // 0-100, already stabilised
      setRiskScore(rollingScore);
      setContentScore(data.content_score ?? 0);
      setVoiceScore(data.voice_score ?? 0);
      setConfidence(data.confidence ?? 0);
      setCategories(data.categories ?? []);
      setInsight(data.insight ?? "");
      setThreatCategory(data.threat_category ?? "");
      setChunkCount(data.chunk_number ?? 0);
      setBackendActive(true);

      // Voice panel data
      if (data.voice_data) {
        latestVoiceRef.current = {
          ...data.voice_data,
          verdict: data.verdict ?? latestVoiceRef.current.verdict,
          verdict_confidence:
            data.verdict_confidence ??
            latestVoiceRef.current.verdict_confidence,
          nearest_matches: data.nearest_matches ?? [],
          acoustic_details: data.acoustic_details ?? {},
          ready: true,
        };
        setVoiceData({ ...latestVoiceRef.current });
      }

      // Append transcript entry from backend (Whisper-transcribed)
      if (data.transcript_entry?.text?.trim()) {
        setTranscriptEntries((prev) => [...prev, data.transcript_entry]);
        allTextRef.current += " " + data.transcript_entry.text;
      }
    } catch (err) {
      console.warn("[LiveMode] Backend chunk failed:", err.message);
      setBackendActive(false);
      // Fallback: use SpeechRecognition text for content scoring only
      const content = analyzeContent(allTextRef.current);
      setContentScore(Math.round(content.score * 10) / 10);
      setRiskScore(Math.round(content.score * 10) / 10);
      setCategories(content.categories);
      setInsight(content.insight);
    }
  }, []);

  /* ── start recording a chunk cycle ─────────────────── */
  const startChunkCycle = useCallback(
    (stream) => {
      const mimeType = MediaRecorder.isTypeSupported(MIME_TYPE)
        ? MIME_TYPE
        : "audio/webm";

      const startRecorderSlice = () => {
        if (!stream.active) return;
        const mr = new MediaRecorder(stream, { mimeType });
        mediaRecorderRef.current = mr;
        const chunks = [];
        mr.ondataavailable = (e) => {
          if (e.data?.size > 0) chunks.push(e.data);
        };
        mr.onstop = () => {
          const blob = new Blob(chunks, { type: mimeType });
          sendChunk(blob);
        };
        mr.start();
      };

      // Record first slice immediately
      startRecorderSlice();

      // Every CHUNK_INTERVAL_MS: stop current recorder (triggers onstop → sendChunk), start next
      chunkIntervalRef.current = setInterval(() => {
        if (mediaRecorderRef.current?.state === "recording") {
          mediaRecorderRef.current.stop();
        }
        // Start next slice only if still recording
        if (stream.active) {
          setTimeout(startRecorderSlice, 50);
        }
      }, CHUNK_INTERVAL_MS);
    },
    [sendChunk],
  );

  /* ── SpeechRecognition for real-time text overlay ───── */
  const startSpeechRecognition = useCallback(() => {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) return;

    const recognition = new SR();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";
    recognitionRef.current = recognition;

    recognition.onresult = (event) => {
      let interim = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const t = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          const text = t.trim();
          if (text.length > 0) {
            // We only use speech recognition for the interim UI overlay
            // (Whisper from backend provides the authoritative transcript)
            setInterimText("");
          }
        } else {
          interim += t;
        }
      }
      setInterimText(interim);
    };

    recognition.onerror = (e) => {
      if (e.error === "not-allowed")
        setMicError("Microphone permission denied.");
      if (e.error === "no-speech" || e.error === "aborted") {
        try {
          recognition.start();
        } catch (_) {}
      }
    };

    recognition.onend = () => {
      if (streamRef.current?.active) {
        try {
          recognition.start();
        } catch (_) {}
      }
    };

    recognition.start();
  }, []);

  /* ── start / stop ────────────────────────────────────── */
  const startRecording = async () => {
    setMicError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 16000,
        },
      });
      streamRef.current = stream;
      sessionIdRef.current = "live-" + Date.now(); // fresh session ID

      startChunkCycle(stream);
      startSpeechRecognition();

      setRecording(true);
      setTimer(0);
      timerRef.current = setInterval(() => setTimer((prev) => prev + 1), 1000);
    } catch (err) {
      setMicError(
        `Microphone access failed: ${err.message}. Please allow microphone permission and try again.`,
      );
    }
  };

  const stopRecording = () => {
    // Stop MediaRecorder — triggers final onstop → sendChunk
    if (mediaRecorderRef.current?.state === "recording") {
      try {
        mediaRecorderRef.current.stop();
      } catch (_) {}
    }
    mediaRecorderRef.current = null;

    if (chunkIntervalRef.current) {
      clearInterval(chunkIntervalRef.current);
      chunkIntervalRef.current = null;
    }

    if (recognitionRef.current) {
      try {
        recognitionRef.current.stop();
      } catch (_) {}
      recognitionRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }

    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    setRecording(false);
    setInterimText("");
  };

  useEffect(() => () => stopRecording(), []);

  const handleReset = () => {
    stopRecording();
    setTranscriptEntries([]);
    setRiskScore(0);
    setContentScore(0);
    setVoiceScore(0);
    setConfidence(0);
    setCategories([]);
    setInsight("");
    setThreatCategory("");
    setVoiceData({});
    setChunkCount(0);
    setTimer(0);
    setInterimText("");
    setBackendActive(false);
    allTextRef.current = "";
    latestVoiceRef.current = {};
  };

  const handleReport = () => {
    const data = JSON.stringify(
      {
        session_id: sessionIdRef.current,
        risk_score: riskScore,
        content_score: contentScore,
        voice_score: voiceScore,
        confidence,
        categories,
        insight,
        threat_category: threatCategory,
        transcript: allTextRef.current.trim(),
        transcript_entries: transcriptEntries,
        voice_data: voiceData,
      },
      null,
      2,
    );
    const blob = new Blob([data], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `scam-report-live-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const hasResults = riskScore > 0 || transcriptEntries.length > 0;

  const displayEntries = interimText
    ? [
        ...transcriptEntries,
        {
          time: getTimeStamp(),
          speaker: "SPEAKER",
          text: interimText + " …",
          flagged: false,
          categories: [],
          interim: true,
        },
      ]
    : transcriptEntries;

  /* ── render ──────────────────────────────────────────── */
  return (
    <>
      <div className="live-controls">
        <div style={{ textAlign: "center" }}>
          <div className="recording-timer">{formatTime(timer)}</div>
          <span className={`record-status ${recording ? "recording" : ""}`}>
            {recording
              ? `Recording — Chunk ${chunkCount}${backendActive ? " · ML Active" : " · Connecting…"}`
              : "Ready to Record"}
          </span>
        </div>
        <button
          className={`record-btn ${recording ? "recording" : ""}`}
          onClick={recording ? stopRecording : startRecording}
          title={recording ? "Stop Recording" : "Start Recording"}
        >
          <div className="inner"></div>
        </button>
      </div>

      {micError && (
        <div
          className="alert-banner"
          style={{ maxWidth: 600, margin: "0 auto 16px" }}
        >
          <div className="alert-icon">🎤</div>
          <div className="alert-content">
            <h4>Notice</h4>
            <p>{micError}</p>
          </div>
        </div>
      )}

      {hasResults && (
        <>
          <AlertBanner score={riskScore} categories={categories} />
          <div className="analysis-grid">
            <div>
              <div className="card" style={{ marginBottom: 24 }}>
                <RiskMeter score={riskScore} />
              </div>
              <VoiceAuthPanel voiceData={voiceData} />
            </div>
            <TranscriptDisplay entries={displayEntries} isLive={recording} />
          </div>
          <div className="bottom-grid">
            <RiskBreakdown
              contentScore={contentScore}
              voiceScore={voiceScore}
              confidence={confidence}
              insight={insight}
              threatCategory={threatCategory}
            />
            <div className="card">
              <SafetyPanel
                score={riskScore}
                onDisconnect={handleReset}
                onReport={handleReport}
                onReset={handleReset}
              />
            </div>
          </div>
        </>
      )}

      {!hasResults && !recording && !micError && (
        <div className="empty-state">
          <div className="icon">🎤</div>
          <h3>Start Live Voice Analysis</h3>
          <p>
            Click the record button to begin real-time scam detection.
            <br />
            Speak or play a call near your microphone — the system will:
            <br />
            <strong>1.</strong> Transcribe your speech using Whisper ML
            <br />
            <strong>2.</strong> Scan for scam keywords as you speak
            <br />
            <strong>3.</strong> Analyze voice patterns for AI-generated speech
            <br />
            <span
              style={{
                fontSize: 11,
                color: "var(--text-muted)",
                marginTop: 8,
                display: "inline-block",
              }}
            >
              Best in Google Chrome · AI model runs on backend · Rolling 3-chunk
              scoring
            </span>
          </p>
        </div>
      )}
    </>
  );
}
