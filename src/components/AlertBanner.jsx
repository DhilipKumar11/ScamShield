import React from 'react';

export default function AlertBanner({ score = 0, categories = [] }) {
  if (score < 30) return null;

  const isHigh = score >= 60;

  return (
    <div className={`alert-banner ${isHigh ? 'high-risk' : ''}`}>
      <div className="alert-icon">
        {isHigh ? '🛡️' : '⚠️'}
      </div>
      <div className="alert-content">
        <h4>
          {isHigh
            ? 'Immediate Security Threat Detected'
            : 'Caution — Suspicious Activity Detected'}
        </h4>
        <p>
          {isHigh
            ? 'Synthetic speech patterns matched with high-urgency financial coercion markers.'
            : 'Some patterns in this conversation may indicate a potential scam attempt.'}
          {categories.length > 0 && ` Categories: ${categories.join(', ')}.`}
        </p>
      </div>
    </div>
  );
}
