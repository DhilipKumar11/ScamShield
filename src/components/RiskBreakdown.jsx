import React from 'react';

export default function RiskBreakdown({ contentScore = 0, voiceScore = 0, confidence = 0, insight = '', threatCategory = '' }) {
  return (
    <div className="card" style={{ animationDelay: '0.2s' }}>
      <div className="card-title">📊 Detailed Risk Assessment</div>
      <div className="card-subtitle">Comprehensive analysis breakdown</div>
      <div className="assessment-scores">
        <div className="assessment-score-item">
          <div className="assessment-score-label">Content Risk</div>
          <div className="assessment-score-value red">{Math.round(contentScore)}/100</div>
        </div>
        <div className="assessment-score-item">
          <div className="assessment-score-label">Voice Suspicion</div>
          <div className="assessment-score-value orange">{Math.round(voiceScore)}/100</div>
        </div>
        <div className="assessment-score-item">
          <div className="assessment-score-label">Global Confidence</div>
          <div className="assessment-score-value purple">{Math.round(confidence)}%</div>
        </div>
      </div>
      {(insight || threatCategory) && (
        <div className="assessment-insight">
          <strong>AI Insight: </strong>
          {insight || 'Analysis in progress...'}
          {threatCategory && (
            <>
              {' '}Threat category: <span className="threat-category">{threatCategory}</span>.
            </>
          )}
        </div>
      )}
    </div>
  );
}
