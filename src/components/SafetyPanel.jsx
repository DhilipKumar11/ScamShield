import React from 'react';

export default function SafetyPanel({ score = 0, onDisconnect, onReport, onReset }) {
  const isHigh = score >= 60;

  return (
    <div className="safety-actions">
      {isHigh && (
        <div className="safety-box">
          <h4>⚠️ Safety Instruction</h4>
          <p>DO NOT share OTP or transfer money. This is a verified fraudulent session.</p>
        </div>
      )}
      {!isHigh && score >= 30 && (
        <div className="safety-box" style={{ background: 'var(--accent-orange)' }}>
          <h4>⚠️ Caution</h4>
          <p>Exercise caution. Verify the caller's identity before sharing any information.</p>
        </div>
      )}
      {score < 30 && score > 0 && (
        <div className="safety-box" style={{ background: 'var(--accent-green)' }}>
          <h4>✅ Safe</h4>
          <p>No significant threat detected. This call appears to be legitimate.</p>
        </div>
      )}
      <div className="action-buttons">
        <button className="btn btn-danger" onClick={onDisconnect}>
          📞 Disconnect
        </button>
        <button className="btn btn-outline" onClick={onReport}>
          📋 Report
        </button>
      </div>
      <button className="reset-btn" onClick={onReset}>
        🔄 Reset Current Session
      </button>
    </div>
  );
}
