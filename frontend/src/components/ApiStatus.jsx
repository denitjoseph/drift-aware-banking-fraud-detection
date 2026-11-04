import React from 'react';

/**
 * Component displaying real-time FastAPI backend connectivity and model readiness.
 */
export default function ApiStatus({ status, healthData, onRefresh, isChecking }) {
  let badgeClass = 'checking';
  let label = 'Checking...';

  if (status === 'connected') {
    badgeClass = 'connected';
    label = 'API Connected';
  } else if (status === 'disconnected') {
    badgeClass = 'disconnected';
    label = 'API Disconnected';
  }

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
      <div className={`api-status-badge ${badgeClass}`} title={healthData?.detail || ''}>
        <span className="status-dot"></span>
        <span>{label}</span>
      </div>

      {status === 'connected' && healthData?.model_name && (
        <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
          Model: <strong>{healthData.model_name}</strong> (Threshold: {healthData.classification_threshold})
        </span>
      )}

      {onRefresh && (
        <button
          onClick={onRefresh}
          disabled={isChecking}
          title="Refresh API Status"
          style={{
            background: 'none',
            border: '1px solid var(--border-color)',
            borderRadius: 'var(--radius-sm)',
            padding: '4px 8px',
            fontSize: '11px',
            cursor: 'pointer',
            color: 'var(--text-secondary)'
          }}
        >
          {isChecking ? '...' : '↻'}
        </button>
      )}
    </div>
  );
}
