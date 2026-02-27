import React from 'react';

const SUSPICIOUS_KEYWORDS = [
  'immediate', 'immediately', 'urgent', 'arrest', 'warrant', 'OTP',
  'payment', 'transfer', 'court', 'police', 'suspend', 'block',
  'verify', 'secure portal', 'bank account', 'legal action',
  'fine', 'penalty', 'deadline', 'expire', 'disconnect',
  // Hindi
  'turant', 'giraftaar', 'bhugtaan', 'khatam',
  // Tamil
  'udanadi', 'kaidhhu', 'panam', 'needhimandram',
];

function highlightText(text) {
  if (!text) return text;
  const regex = new RegExp(`(${SUSPICIOUS_KEYWORDS.join('|')})`, 'gi');
  const parts = text.split(regex);
  return parts.map((part, i) => {
    if (SUSPICIOUS_KEYWORDS.some(k => k.toLowerCase() === part.toLowerCase())) {
      return <span key={i} className="highlight">{part}</span>;
    }
    return part;
  });
}

export default function TranscriptDisplay({ entries = [], isLive = false }) {
  const threats = [];
  const threatSet = new Set();
  
  entries.forEach(e => {
    if (e.categories) {
      e.categories.forEach(c => {
        if (!threatSet.has(c)) {
          threatSet.add(c);
          threats.push(c);
        }
      });
    }
  });

  return (
    <div className="card transcript-panel">
      <div className="transcript-header">
        <div className="card-title">📋 Live Analysis Transcript</div>
        {isLive && (
          <span className="live-badge">
            <span className="dot"></span>
            Live Stream
          </span>
        )}
      </div>
      <div className="transcript-body">
        {entries.length === 0 && (
          <div className="empty-state">
            <div className="icon">🎤</div>
            <h3>No transcript yet</h3>
            <p>Upload audio or start recording to see the live transcript analysis.</p>
          </div>
        )}
        {entries.map((entry, i) => (
          <div
            key={i}
            className={`transcript-entry ${entry.flagged ? 'flagged' : ''}`}
          >
            <span className="transcript-time">{entry.time || '--:--:--'}</span>
            <div className="transcript-content">
              <div className="transcript-speaker">{entry.speaker || 'SPEAKER'}</div>
              <div className="transcript-text">{highlightText(entry.text)}</div>
            </div>
          </div>
        ))}
      </div>
      {threats.length > 0 && (
        <div className="threat-tags">
          {threats.map((t, i) => (
            <span className="threat-tag" key={i}>
              <span className="icon">⚠️</span>
              {t}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
