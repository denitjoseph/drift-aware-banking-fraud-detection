import React from 'react';

/**
 * Component to render the machine learning prediction outcome returned by the FastAPI backend.
 */
export default function PredictionResult({ result, isLoading, error }) {
  if (isLoading) {
    return (
      <div className="result-card result-card-placeholder">
        <div className="spinner" style={{ borderColor: 'rgba(37, 99, 235, 0.2)', borderTopColor: '#2563eb', width: '32px', height: '32px', marginBottom: '12px' }}></div>
        <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>Evaluating Transaction...</div>
        <div style={{ fontSize: '13px', color: 'var(--text-muted)', marginTop: '4px' }}>
          Sending 30 features to Scikit-learn Random Forest pipeline
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="result-card" style={{ borderColor: 'var(--color-danger-border)' }}>
        <div className="alert alert-danger" style={{ marginBottom: 0 }}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
            <circle cx="12" cy="12" r="10"/>
            <line x1="12" x2="12" y1="8" y2="12"/>
            <line x1="12" x2="12.01" y1="16" y2="16"/>
          </svg>
          <div>
            <strong>Inference Request Failed</strong>
            <div style={{ marginTop: '4px', fontSize: '12px' }}>{error}</div>
          </div>
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="result-card result-card-placeholder">
        <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{ margin: '0 auto 12px', color: '#94a3b8' }}>
          <rect width="20" height="14" x="2" y="5" rx="2"/>
          <line x1="2" x2="22" y1="10" y2="10"/>
        </svg>
        <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>No Prediction Yet</div>
        <div style={{ fontSize: '13px', color: 'var(--text-muted)', marginTop: '4px' }}>
          Fill in the transaction features and click "Predict Transaction", or choose one of the sample presets.
        </div>
      </div>
    );
  }

  const isFraud = result.prediction === 'FRAUD';
  const probPercent = (result.fraud_probability * 100).toFixed(2);
  const threshPercent = (result.classification_threshold * 100).toFixed(2);

  return (
    <div className={`result-card ${isFraud ? 'fraud' : 'legitimate'}`}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: '12px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)' }}>
          Model Inference Decision
        </span>
        <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
          Phase 2 Pipeline Model
        </span>
      </div>

      <div style={{ marginTop: '14px', marginBottom: '16px' }}>
        <div className={`result-badge ${isFraud ? 'fraud' : 'legitimate'}`}>
          {isFraud ? (
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>
              <line x1="12" x2="12" y1="9" y2="13"/>
              <line x1="12" x2="12.01" y1="17" y2="17"/>
            </svg>
          ) : (
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
              <polyline points="22 4 12 14.01 9 11.01"/>
            </svg>
          )}
          {result.prediction}
        </div>
      </div>

      <div className="result-metrics">
        <div className="metric-box">
          <div className="metric-label">Fraud Probability</div>
          <div className="metric-value" style={{ color: isFraud ? 'var(--color-danger)' : 'var(--color-success)' }}>
            {probPercent}%
          </div>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>
            Score: {result.fraud_probability}
          </div>
        </div>

        <div className="metric-box">
          <div className="metric-label">Classification Threshold</div>
          <div className="metric-value">{threshPercent}%</div>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>
            Cutoff: {result.classification_threshold}
          </div>
        </div>
      </div>

      <div className="result-explanation">
        <strong>Decision Logic: </strong>
        {isFraud ? (
          <span>
            The model assessed the fraud probability ({result.fraud_probability}) as being{' '}
            <strong>greater than or equal to</strong> the configured cutoff threshold (
            {result.classification_threshold}). The transaction is flagged as high-risk <strong>FRAUD</strong>.
          </span>
        ) : (
          <span>
            The model assessed the fraud probability ({result.fraud_probability}) as being{' '}
            <strong>below</strong> the classification threshold ({result.classification_threshold}). The transaction is categorized as <strong>LEGITIMATE</strong>.
          </span>
        )}
      </div>
    </div>
  );
}
