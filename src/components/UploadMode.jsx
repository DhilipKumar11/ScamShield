import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  analyzeContent,
  analyzeAudioBuffer,
  computeRisk,
  decodeAudioFile,
  compareWithDataset,
} from '../audioEngine';
import RiskMeter from './RiskMeter';
import VoiceAuthPanel from './VoiceAuthPanel';
import TranscriptDisplay from './TranscriptDisplay';
import SafetyPanel from './SafetyPanel';

// ─── Dataset Match Panel ──────────────────────────────────────────────────────
function DatasetMatchPanel({ datasetResult }) {
  if (!datasetResult || !datasetResult.ready) return null;
  const { datasetScore, topScamMatches, topNonScamMatches, scamSim, nonScamSim } = datasetResult;
  const color = datasetScore >= 70 ? '#ef4444' : datasetScore >= 40 ? '#f97316' : '#22c55e';
  return (
    <div className="dataset-match-panel">
      <h3 className="dm-title">📊 Dataset Pattern Comparison</h3>
      <div className="dm-score-row">
        <div className="dm-score-badge" style={{ borderColor: color, color }}>
          <span className="dm-score-num">{datasetScore}%</span>
          <span className="dm-score-label">Scam Pattern Match</span>
        </div>
        <div className="dm-sims">
          <div className="dm-sim-item">
            <span className="dm-sim-dot" style={{ background: '#ef4444' }} />
            Scam similarity: <strong>{scamSim}%</strong>
          </div>
          <div className="dm-sim-item">
            <span className="dm-sim-dot" style={{ background: '#22c55e' }} />
            Legit similarity: <strong>{nonScamSim}%</strong>
          </div>
        </div>
      </div>
      {topScamMatches.filter(m => m.score > 0.02).length > 0 && (
        <div className="dm-matches">
          <p className="dm-matches-label">⚠ Closest scam patterns:</p>
          {topScamMatches.filter(m => m.score > 0.02).slice(0, 3).map((m, i) => (
            <div key={i} className="dm-match-item dm-scam">
              <span className="dm-match-score">{(m.score * 100).toFixed(0)}%</span>
              <span className="dm-match-text">{m.text}</span>
            </div>
          ))}
        </div>
      )}
      {datasetScore < 60 && topNonScamMatches.filter(m => m.score > 0.02).length > 0 && (
        <div className="dm-matches">
          <p className="dm-matches-label">✅ Closest legitimate patterns:</p>
          {topNonScamMatches.filter(m => m.score > 0.02).slice(0, 2).map((m, i) => (
            <div key={i} className="dm-match-item dm-safe">
              <span className="dm-match-score">{(m.score * 100).toFixed(0)}%</span>
              <span className="dm-match-text">{m.text}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Voice Modulation Detail Panel ────────────────────────────────────────────
function VoiceModulationPanel({ voiceResult }) {
  if (!voiceResult || !voiceResult._raw || !voiceResult._raw.scores) return null;
  const raw = voiceResult._raw;
  const scores = raw.scores;

  const signals = [
    {
      icon: '🎼', name: 'Pitch Variation (CV)',
      detail: `CV = ${(raw.pitch_cv * 100).toFixed(1)}%  •  Human > 12%, AI < 6%`,
      value: Math.round(scores.pitchCVScore * 100),
      bar: scores.pitchCVScore,
    },
    {
      icon: '🔬', name: 'Pitch Jitter (cycle-to-cycle)',
      detail: `Jitter = ${(raw.jitter * 100).toFixed(2)}%  •  Human 0.5-1.5%, AI < 0.1%`,
      value: Math.round(scores.jitterScore * 100),
      bar: scores.jitterScore,
    },
    {
      icon: '📡', name: 'Periodicity (ACF Strength)',
      detail: `ACF = ${(raw.acf_mean).toFixed(3)}  •  Human < 0.65, AI > 0.75`,
      value: Math.round(scores.acfScore * 100),
      bar: scores.acfScore,
    },
    {
      icon: '📊', name: 'Shimmer (amplitude variation)',
      detail: `Shimmer = ${(raw.shimmer * 100).toFixed(1)}%  •  Human 3-8%, AI < 1%`,
      value: Math.round(scores.shimmerScore * 100),
      bar: scores.shimmerScore,
    },
    {
      icon: '🌊', name: 'Spectral Flatness Stability',
      detail: `Flatness CV = ${raw.flatness_cv.toFixed(2)}  •  Human > 2.5, AI < 1.5`,
      value: Math.round(scores.flatScore * 100),
      bar: scores.flatScore,
    },
  ];

  return (
    <div className="card voice-modulation-panel">
      <div className="card-title">🧬 Voice Modulation Analysis</div>
      <div className="card-subtitle">
        Each signal independently detects AI characteristics — higher = more AI-like
      </div>
      {signals.map((s, i) => {
        const barColor = s.value >= 60 ? '#ef4444' : s.value >= 35 ? '#f97316' : '#22c55e';
        return (
          <div key={i} className="mod-signal">
            <div className="mod-header">
              <span className="mod-icon">{s.icon}</span>
              <span className="mod-name">{s.name}</span>
              <span className="mod-value" style={{ color: barColor }}>{s.value}%</span>
            </div>
            <div className="mod-bar-track">
              <div className="mod-bar-fill" style={{ width: `${s.value}%`, background: barColor }} />
            </div>
            <div className="mod-detail">{s.detail}</div>
          </div>
        );
      })}
      <div className="mod-knn-note">
        🤖 KNN classifier score: <strong>{raw.knn_prob?.toFixed(1) ?? 'N/A'}%</strong> (from 200 audio reference samples)
      </div>
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────
export default function UploadMode() {
  const [file,           setFile]          = useState(null);
  const [loading,        setLoading]       = useState(false);
  const [loadingStatus,  setLoadingStatus] = useState('');
  const [result,         setResult]        = useState(null);
  const [error,          setError]         = useState('');
  const [dragOver,       setDragOver]      = useState(false);

  // Dataset
  const [scamLines,   setScamLines]     = useState([]);
  const [nonScamLines,setNonScamLines]  = useState([]);
  const [datasetReady,setDatasetReady]  = useState(false);

  const inputRef = useRef();

  // ── Load datasets once ────────────────────────────────────────────────────
  useEffect(() => {
    Promise.all([
      fetch('/datasets/English_Scam.txt').then(r => r.text()),
      fetch('/datasets/English_NonScam.txt').then(r => r.text()),
    ]).then(([s, n]) => {
      setScamLines(s.split('\n').filter(l => l.trim().length > 15));
      setNonScamLines(n.split('\n').filter(l => l.trim().length > 15));
      setDatasetReady(true);
    }).catch(e => console.warn('[ScamShield] Dataset load failed:', e));
  }, []);

  // ── File picker ────────────────────────────────────────────────────────────
  const handleFile = useCallback((f) => {
    if (!f) return;
    const ok = f.type?.startsWith('audio/') || f.name.match(/\.(wav|mp3|ogg|webm|m4a|flac|aac)$/i);
    if (!ok) { setError('Unsupported format. Please upload WAV, MP3, OGG, WEBM, M4A, FLAC, or AAC.'); return; }
    setFile(f); setResult(null); setError('');
  }, []);

  const handleDrop = (e) => {
    e.preventDefault(); setDragOver(false);
    handleFile(e.dataTransfer.files[0]);
  };

  // ── Transcribe audio by playing through SpeechRecognition API ──────────────
  const transcribeAudio = (audioFile) => {
    return new Promise((resolve) => {
      const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SR) { resolve(''); return; }

      const url = URL.createObjectURL(audioFile);
      const audio = new Audio(url);
      audio.volume = 0.01; // near-silent playback

      const recognition = new SR();
      recognition.continuous = true;
      recognition.interimResults = false;
      recognition.lang = 'en-US';
      recognition.maxAlternatives = 1;

      let fullText = '';
      let timeout;

      recognition.onresult = (event) => {
        for (let i = event.resultIndex; i < event.results.length; i++) {
          if (event.results[i].isFinal) {
            fullText += ' ' + event.results[i][0].transcript;
          }
        }
      };

      const finish = () => {
        clearTimeout(timeout);
        try { recognition.stop(); } catch (_) {}
        try { audio.pause(); } catch (_) {}
        URL.revokeObjectURL(url);
        resolve(fullText.trim());
      };

      recognition.onerror = () => finish();
      recognition.onend   = () => finish();
      audio.onended       = () => { setTimeout(finish, 1500); };
      audio.onerror       = () => finish();

      // Safety timeout — max 90 seconds of transcription
      timeout = setTimeout(finish, 90000);

      audio.play().then(() => {
        recognition.start();
      }).catch(() => finish());
    });
  };

  // ── MAIN ANALYSIS — full pipeline ─────────────────────────────────────────
  const processFile = async () => {
    if (!file) return;
    setLoading(true); setResult(null); setError('');

    try {
      // Step 1: Decode audio (for frontend modulation panel)
      setLoadingStatus('🔬 Decoding audio…');
      const audioBuffer = await decodeAudioFile(file);

      // Step 2: JS heuristic voice analysis (for modulation panel visuals)
      setLoadingStatus('🤖 Analysing voice modulation, pitch, jitter, shimmer…');
      const voiceResJS = analyzeAudioBuffer(audioBuffer);

      // Step 3: Send to backend ML model (SVM trained on Kaggle dataset, AUC 0.97)
      setLoadingStatus('🧠 Running ML voice authenticity model…');
      let backendResult = null;
      try {
        const formData = new FormData();
        formData.append('file', file);
        const resp = await fetch('http://localhost:8000/analyze_audio', {
          method: 'POST',
          body: formData,
        });
        if (resp.ok) backendResult = await resp.json();
      } catch (e) {
        console.warn('[ScamShield] Backend unavailable — using JS heuristic fallback:', e.message);
      }

      // Step 4: Pick voice score — ML preferred, JS fallback
      const mlVoiceScore    = backendResult?.voice_score ?? null;
      const finalVoiceScore = mlVoiceScore !== null ? mlVoiceScore : voiceResJS.synthetic_probability;
      const voiceRes = {
        ...voiceResJS,
        synthetic_probability: finalVoiceScore,
        method:              mlVoiceScore !== null ? 'ensemble' : 'heuristic',
        model_auc:           backendResult ? '0.97' : null,
        // ── Ensemble verdict fields from backend ──────────────────
        verdict:             backendResult?.verdict ?? 'UNCERTAIN',
        verdict_confidence:  backendResult?.verdict_confidence ?? 0,
        ml_score:            backendResult?.ml_score ?? null,
        knn_score:           backendResult?.knn_score ?? null,
        nearest_matches:     backendResult?.nearest_matches ?? [],
        acoustic_details:    backendResult?.acoustic_details ?? {},
        // Also spread backend voice_data for VoiceAuthPanel panel fields
        ...(backendResult?.voice_data ?? {}),
      };

      // Step 5: Transcription — backend first, then browser SpeechRecognition
      let transcript = backendResult?.transcript || '';
      if (!transcript) {
        setLoadingStatus('🎙️ Transcribing audio to text…');
        transcript = await transcribeAudio(file);
      }

      // Step 6: Content analysis
      let contentResult  = null;
      let datasetResult  = null;
      let insight        = backendResult?.insight || '';
      let threatCategory = backendResult?.threat_category || '';
      let contentScore   = backendResult?.content_score ?? 0;

      if (transcript && transcript.length > 10) {
        setLoadingStatus('📋 Scanning transcript for scam keywords…');
        contentResult = analyzeContent(transcript);
        if (!backendResult) {
          contentScore   = contentResult.score;
          insight        = contentResult.insight;
          threatCategory = contentResult.threatCategory;
        }
        if (datasetReady && scamLines.length > 0 && !backendResult) {
          setLoadingStatus('📊 Comparing with scam/legit dataset patterns…');
          datasetResult = compareWithDataset(transcript, scamLines, nonScamLines);
          if (datasetResult.ready)
            contentScore = Math.round(contentResult.score * 0.6 + datasetResult.datasetScore * 0.4);
        }
      } else if (!insight) {
        insight = 'Auto-transcription returned no text. Voice-only analysis performed using 6 acoustic signals.';
      }

      // Step 7: Risk fusion
      const risk = computeRisk(contentScore, finalVoiceScore);
      if (backendResult?.risk_score != null) {
        risk.finalScore = backendResult.risk_score;
        risk.confidence = backendResult.confidence ?? risk.confidence;
      }

      // Transcript entries
      const now = new Date().toLocaleTimeString('en-US', {
        hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit'
      });
      const entries = backendResult?.transcript_entries ||
        (transcript
          ? transcript.split(/(?<=[.!?])\s+/).filter(s => s.length > 5).map(s => {
              const seg = analyzeContent(s);
              return { time: now, speaker: 'SPEAKER', text: s,
                       flagged: seg.score > 10, categories: seg.score > 10 ? seg.categories : [] };
            })
          : []);

      setResult({
        risk, voiceResult: voiceRes, contentResult, datasetResult,
        entries, transcript: transcript || '',
        insight, threatCategory, contentScore,
        mlVoiceScore, backendUsed: !!backendResult,
      });
    } catch (err) {
      console.error(err);
      setError('Analysis failed: ' + (err.message || err));
    } finally {
      setLoading(false);
      setLoadingStatus('');
    }
  };


  const reset = () => {
    setFile(null); setResult(null); setError('');
  };

  // Download a JSON report
  const downloadReport = () => {
    if (!result) return;
    const report = {
      session_id:     'upload-' + Date.now(),
      file_name:      file?.name ?? 'unknown',
      risk_score:     result.risk.finalScore,
      risk_level:     result.risk.level,
      confidence:     result.risk.confidence,
      voice_score:    result.voiceResult?.synthetic_probability ?? 0,
      content_score:  result.contentScore ?? 0,
      dataset_score:  result.datasetResult?.datasetScore ?? null,
      transcript:     result.transcript,
      categories:     result.contentResult?.categories ?? [],
      voice_data:     result.voiceResult ?? {},
    };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url; a.download = `scam-report-${Date.now()}.json`; a.click();
    URL.revokeObjectURL(url);
  };

  // ────────────────────────────────────────────────────────────────────────────
  return (
    <div className="upload-mode">

      {/* ── Upload zone ── */}
      {!result && (
        <div
          className={`upload-zone ${dragOver ? 'drag-over' : ''} ${file ? 'has-file' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => !file && inputRef.current?.click()}
        >
          <input
            ref={inputRef} type="file" accept="audio/*"
            style={{ display: 'none' }}
            onChange={(e) => handleFile(e.target.files[0])}
          />
          {file ? (
            <div className="file-selected">
              <div className="file-icon">🎵</div>
              <div className="file-name">{file.name}</div>
              <div className="file-size">{(file.size / 1024 / 1024).toFixed(2)} MB</div>
              <button
                className="change-file-btn"
                onClick={(e) => { e.stopPropagation(); inputRef.current?.click(); }}
              >Change file</button>
            </div>
          ) : (
            <div className="upload-prompt">
              <div className="upload-icon">📁</div>
              <p className="upload-text">Drop audio file here or click to browse</p>
              <p className="upload-sub">WAV · MP3 · OGG · WEBM · M4A · FLAC · AAC</p>
            </div>
          )}
        </div>
      )}

      {error && <div className="upload-error">{error}</div>}

      {/* ── Dataset badge ── */}
      {!result && !loading && (
        <div className={`dataset-status-badge ${datasetReady ? 'ready' : 'loading'}`}>
          {datasetReady
            ? `✅ ${scamLines.length} scam + ${nonScamLines.length} legit patterns loaded`
            : '⏳ Loading detection dataset…'}
        </div>
      )}

      {/* ── Analyse button ── */}
      {file && !result && !loading && (
        <button className="analyse-btn" onClick={processFile}>
          🔍 Full Analysis (Voice + Transcription + Dataset)
        </button>
      )}

      {/* ── Loading ── */}
      {loading && (
        <div className="loading-state">
          <div className="loading-spinner" />
          <p className="loading-status">{loadingStatus}</p>
        </div>
      )}

      {/* ── RESULTS ── */}
      {result && (
        <div className="results-section">
          {/* Header row */}
          <div className="rs-header">
            <div className="rs-filename">📁 {file?.name}</div>
            <button className="reset-btn" onClick={reset}>⬅ New Upload</button>
          </div>

          {/* ① Big threat score — top of page */}
          <RiskMeter
            score={result.risk.finalScore}
            insight={result.insight || ''}
            threatCategory={result.threatCategory || ''}
            categories={result.categories || []}
          />

          {/* ② Voice authenticity — single line + expandable advanced */}
          <VoiceAuthPanel voiceData={result.voiceResult} />

          {/* ③ Score breakdown chips */}
          <div className="rs-score-row">
            <div className="rs-chip">
              <span className="rs-chip-label">Content Risk</span>
              <span className="rs-chip-val" style={{ color: (result.contentScore ?? 0) >= 60 ? '#ef4444' : (result.contentScore ?? 0) >= 30 ? '#f59e0b' : '#22c55e' }}>
                {Math.round(result.contentScore ?? 0)}
              </span>
            </div>
            <div className="rs-chip">
              <span className="rs-chip-label">Voice Score</span>
              <span className="rs-chip-val" style={{ color: (result.voiceResult?.synthetic_probability ?? 0) >= 58 ? '#ef4444' : '#22c55e' }}>
                {Math.round(result.voiceResult?.synthetic_probability ?? 0)}
              </span>
            </div>
            <div className="rs-chip">
              <span className="rs-chip-label">Confidence</span>
              <span className="rs-chip-val" style={{ color: '#a78bfa' }}>
                {Math.round(result.risk.confidence ?? 0)}%
              </span>
            </div>
            {result.backendUsed && (
              <div className="rs-chip rs-chip-ml">
                <span className="rs-chip-label">Engine</span>
                <span className="rs-chip-val" style={{ color: '#4ade80', fontSize: 11 }}>
                  SVM · AUC 0.97
                </span>
              </div>
            )}
          </div>

          {/* ④ Transcript with highlighted phrases */}
          {result.transcript ? (
            <TranscriptDisplay entries={result.entries} />
          ) : (
            <div className="transcript-prompt-card">
              <div className="tp-icon">🎙️</div>
              <div className="tp-body">
                <h4>No transcript detected</h4>
                <p>Auto-transcription returned no text. Analysis is based on voice modulation signals only.</p>
              </div>
            </div>
          )}

          {/* ⑤ Safety actions */}
          <SafetyPanel
            score={result.risk.finalScore}
            onDisconnect={reset}
            onReport={downloadReport}
            onReset={reset}
          />
        </div>
      )}
    </div>
  );
}
