import React, { useState } from 'react';

export default function VoiceAuthPanel({ voiceData = {} }) {
  const [expanded, setExpanded] = useState(false);

  const verdict  = voiceData.verdict ?? 'UNCERTAIN';
  const conf     = voiceData.verdict_confidence ?? 0;
  const synProb  = voiceData.synthetic_probability ?? 50;
  const ad       = voiceData.acoustic_details ?? {};
  const matches  = voiceData.nearest_matches ?? [];

  const isAI     = verdict === 'AI_VOICE';
  const isHuman  = verdict === 'HUMAN_VOICE';

  const label    = isAI    ? 'Likely AI-Generated Voice'
                 : isHuman ? 'Likely Human Voice'
                 :           'Voice — Inconclusive';

  const color    = isAI    ? '#ef4444'
                 : isHuman ? '#22c55e'
                 :           '#f59e0b';

  const bg       = isAI    ? 'rgba(239,68,68,0.08)'
                 : isHuman ? 'rgba(34,197,94,0.08)'
                 :           'rgba(245,158,11,0.08)';

  const icon     = isAI ? '🤖' : isHuman ? '🧑' : '❓';

  // Acoustic detail rows (shown in advanced section)
  const adRows = [
    { label: 'Pitch Variation',    value: ad.pitch_cv_pct,          unit: '%',   aiLow: 8,   humHigh: 14,  invert: true  },
    { label: 'Jitter',             value: ad.jitter_pct,            unit: '%',   aiLow: 0.3, humHigh: 0.7, invert: true  },
    { label: 'Shimmer',            value: ad.shimmer_pct,           unit: '%',   aiLow: 3,   humHigh: 6,   invert: true  },
    { label: 'HNR',                value: ad.hnr_db,                unit: ' dB', aiLow: 20,  humHigh: 15,  invert: false },
    { label: 'MFCC Δ Smoothness',  value: ad.mfcc_delta_std,        unit: '',    aiLow: 3,   humHigh: 6,   invert: true  },
    { label: 'ACF Periodicity',    value: ad.acf_periodicity != null ? ad.acf_periodicity * 100 : null, unit: '%', aiLow: 55, humHigh: 72, invert: false },
    { label: 'Voiced Ratio',       value: ad.voiced_ratio_pct,      unit: '%',   aiLow: 0,   humHigh: 0,   invert: true  },
  ];

  return (
    <div className="rs-voice-card" style={{ background: bg, borderColor: color + '44' }}>
      {/* Primary one-liner */}
      <div className="rs-voice-primary">
        <span className="rs-voice-icon">{icon}</span>
        <div>
          <div className="rs-voice-label" style={{ color }}>{label}</div>
          <div className="rs-voice-conf">
            {conf > 0 ? `${conf.toFixed(0)}% confidence` : `Voice score: ${synProb.toFixed(0)}%`}
            {voiceData.method === 'ensemble' && <span className="rs-ml-badge">ML Ensemble</span>}
          </div>
        </div>
      </div>

      {/* Advanced toggle */}
      <button className="rs-advanced-btn" onClick={() => setExpanded(v => !v)}>
        {expanded ? '▲ Hide Advanced Analysis' : '▼ Advanced Analysis'}
      </button>

      {expanded && (
        <div className="rs-advanced-body">
          {/* Acoustic metrics table */}
          {Object.keys(ad).length > 0 && (
            <div className="rs-adv-section">
              <div className="rs-adv-title">Acoustic Signals</div>
              {adRows.filter(r => r.value != null).map((row, i) => {
                const num = parseFloat(row.value) || 0;
                let tag = 'borderline';
                if (row.invert) {
                  tag = num <= row.aiLow ? 'ai-pattern' : num >= row.humHigh ? 'human-pattern' : 'borderline';
                } else {
                  tag = num >= row.aiLow ? 'ai-pattern' : num <= row.humHigh ? 'human-pattern' : 'borderline';
                }
                const tagLabel = { 'ai-pattern': 'AI', 'human-pattern': 'Human', borderline: '—' }[tag];
                const tagColor = { 'ai-pattern': '#ef4444', 'human-pattern': '#22c55e', borderline: '#888' }[tag];
                return (
                  <div key={i} className="rs-adv-row">
                    <span className="rs-adv-name">{row.label}</span>
                    <span className="rs-adv-val">{num.toFixed(2)}{row.unit}</span>
                    <span className="rs-adv-tag" style={{ color: tagColor }}>{tagLabel}</span>
                  </div>
                );
              })}
              {ad.pitch_mean_hz && (
                <div className="rs-adv-meta">
                  Pitch: {ad.pitch_mean_hz} Hz · Duration: {ad.duration_s}s · Voiced: {ad.voiced_ratio_pct}%
                </div>
              )}
            </div>
          )}

          {/* KNN nearest matches */}
          {matches.length > 0 && (
            <div className="rs-adv-section">
              <div className="rs-adv-title">Nearest Dataset Matches</div>
              <table className="rs-match-table">
                <thead>
                  <tr>
                    <th>#</th><th>File</th><th>Label</th><th>Match</th>
                  </tr>
                </thead>
                <tbody>
                  {matches.map((m, i) => (
                    <tr key={i}>
                      <td>{m.rank}</td>
                      <td className="rs-match-file">{m.file}</td>
                      <td>
                        <span style={{ color: m.label === 'AI' ? '#ef4444' : '#22c55e', fontWeight: 600 }}>
                          {m.label === 'AI' ? '🤖 AI' : '🧑 Human'}
                        </span>
                      </td>
                      <td>{m.similarity?.toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="rs-adv-meta">78-dim cosine distance · {matches.length} nearest of 52 dataset files</div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
