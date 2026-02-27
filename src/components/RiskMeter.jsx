import React from 'react';

const RISK_LEVELS = [
  { max: 30,  label: 'Low Risk',               cls: 'low',    color: '#22c55e', bg: 'rgba(34,197,94,0.1)',  border: 'rgba(34,197,94,0.25)' },
  { max: 60,  label: 'Medium Risk',             cls: 'medium', color: '#f59e0b', bg: 'rgba(245,158,11,0.1)', border: 'rgba(245,158,11,0.25)' },
  { max: 101, label: 'High Risk — Scam Detected', cls: 'high', color: '#ef4444', bg: 'rgba(239,68,68,0.1)',  border: 'rgba(239,68,68,0.25)' },
];

function getLevel(score) {
  return RISK_LEVELS.find(l => score < l.max) || RISK_LEVELS[2];
}

export default function RiskMeter({ score = 0, insight = '', threatCategory = '', categories = [] }) {
  const s      = Math.min(100, Math.max(0, Math.round(score)));
  const level  = getLevel(s);
  const pct    = s / 100;

  // Arc geometry (semi-circle)
  const R   = 80;
  const circ = Math.PI * R;
  const off  = circ - pct * circ;

  // Detected signals line
  const signals = [];
  if (threatCategory) signals.push(threatCategory);
  categories.forEach(c => { if (!signals.includes(c)) signals.push(c); });
  const signalLine = signals.length
    ? `Detected: ${signals.join(' + ')}`
    : insight
      ? insight.split('.')[0]
      : null;

  return (
    <div className="rs-threat-card" style={{ background: level.bg, border: `1.5px solid ${level.border}` }}>
      {/* Score arc */}
      <div className="rs-arc-wrap">
        <svg viewBox="0 0 200 120" className="rs-arc-svg">
          <path d="M 20 110 A 80 80 0 0 1 180 110" fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="14" strokeLinecap="round"/>
          <path d="M 20 110 A 80 80 0 0 1 180 110" fill="none"
            stroke={level.color} strokeWidth="14" strokeLinecap="round"
            strokeDasharray={circ} strokeDashoffset={off}
            style={{ transition: 'stroke-dashoffset 0.8s ease' }}
          />
          <text x="100" y="88" textAnchor="middle" fontSize="46" fontWeight="900" fill={level.color} fontFamily="inherit">{s}</text>
          <text x="100" y="110" textAnchor="middle" fontSize="10" fill="rgba(255,255,255,0.45)" letterSpacing="2" fontFamily="inherit">THREAT SCORE</text>
        </svg>
      </div>

      {/* Decision label */}
      <div className="rs-verdict" style={{ color: level.color }}>{level.label}</div>

      {/* Short explanation */}
      {signalLine && (
        <div className="rs-signal-line">{signalLine}</div>
      )}
    </div>
  );
}
