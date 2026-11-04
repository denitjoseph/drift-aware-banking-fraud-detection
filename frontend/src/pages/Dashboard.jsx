import React from 'react';
import StatCard from '../components/StatCard';

/**
 * Main Overview Dashboard Page.
 */
export default function Dashboard({ apiStatus, healthData, onNavigateToPrediction }) {
  const isConnected = apiStatus === 'connected';

  return (
    <div>
      <div className="page-header">
        <h2 className="page-title">MLOps Monitoring & Overview</h2>
        <p className="page-description">
          Real-time health of the inference microservice, model threshold parameters, and dataset benchmark baseline.
        </p>
      </div>

      {/* 4 Stat Cards */}
      <div className="stats-grid">
        <StatCard
          title="Total Transactions"
          value="283,726"
          subtitle="Processed training & validation records"
          tag="Benchmark Data"
          tagType="demo"
          icon={
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>
            </svg>
          }
        />

        <StatCard
          title="Fraudulent Transactions"
          value="473"
          subtitle="0.17% verified fraud occurrences"
          tag="Benchmark Data"
          tagType="demo"
          icon={
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#dc2626" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>
              <line x1="12" x2="12" y1="9" y2="13"/>
              <line x1="12" x2="12.01" y1="17" y2="17"/>
            </svg>
          }
        />

        <StatCard
          title="Legitimate Transactions"
          value="283,253"
          subtitle="99.83% genuine customer transfers"
          tag="Benchmark Data"
          tagType="demo"
          icon={
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#16a34a" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
              <polyline points="22 4 12 14.01 9 11.01"/>
            </svg>
          }
        />

        <StatCard
          title="Inference API Status"
          value={isConnected ? 'Connected' : 'Offline'}
          subtitle={
            isConnected
              ? `Model: ${healthData?.model_name || 'Random Forest'}`
              : 'FastAPI backend unavailable'
          }
          tag={isConnected ? 'Live Microservice' : 'Offline'}
          tagType={isConnected ? 'live' : 'demo'}
          icon={
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke={isConnected ? '#16a34a' : '#dc2626'} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
            </svg>
          }
        />
      </div>

      {/* Operational Information Grids */}
      <div className="info-grid">
        <div className="card">
          <h3 className="card-title">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10"/>
              <line x1="12" x2="16" y1="12" y2="12"/>
              <line x1="12" x2="12" y1="8" y2="12"/>
            </svg>
            System & Architecture Status
          </h3>
          <p className="card-text">
            This end-to-end MLOps platform serves an optimized Scikit-learn Random Forest model through a production-grade FastAPI microservice. Preprocessing transformations (StandardScaler on Time and Amount) and the classifier are packaged into a unified pipeline artifact to eliminate training-serving skew.
          </p>

          <table className="data-table">
            <thead>
              <tr>
                <th>Component</th>
                <th>Technology</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Model Serving API</td>
                <td>FastAPI + Uvicorn</td>
                <td><span style={{ color: isConnected ? 'var(--color-success)' : 'var(--color-danger)', fontWeight: 600 }}>{isConnected ? 'Active & Ready' : 'Offline'}</span></td>
              </tr>
              <tr>
                <td>Containerization</td>
                <td>Docker (python:3.13-slim)</td>
                <td>Verified (Phase 3)</td>
              </tr>
              <tr>
                <td>Machine Learning Model</td>
                <td>Scikit-learn Pipeline (Random Forest)</td>
                <td>Refitted 80% Subset (Phase 2)</td>
              </tr>
              <tr>
                <td>Decision Threshold</td>
                <td>Tuned Cutoff ({healthData?.classification_threshold ?? '0.40'})</td>
                <td>Configured via configs/model_config.json</td>
              </tr>
              <tr>
                <td>Cluster Orchestration</td>
                <td>Kubernetes / AWS EKS</td>
                <td><span style={{ color: 'var(--text-muted)' }}>Planned (Phase 5)</span></td>
              </tr>
            </tbody>
          </table>

          <div style={{ marginTop: '20px' }}>
            <button
              className="btn-primary"
              onClick={onNavigateToPrediction}
            >
              Test Transaction Prediction &rarr;
            </button>
          </div>
        </div>

        <div className="card">
          <h3 className="card-title">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 20h9"/>
              <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/>
            </svg>
            Threshold Tuning Summary
          </h3>
          <p className="card-text" style={{ fontSize: '13px' }}>
            In banking fraud detection, false negatives (missed frauds) carry severe financial consequences compared to false alerts.
          </p>
          <div style={{ background: '#f8fafc', padding: '14px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)', marginBottom: '14px' }}>
            <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Current Operating Point:</div>
            <div style={{ fontSize: '20px', fontWeight: 700, color: 'var(--color-primary)' }}>Threshold = {healthData?.classification_threshold ?? '0.40'}</div>
            <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
              Selected during Phase 2 validation to optimize F1 Score and Precision-Recall balance.
            </div>
          </div>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            Transactions scoring &ge; {healthData?.classification_threshold ?? 0.40} are flagged as <strong>FRAUD</strong>.
          </div>
        </div>
      </div>
    </div>
  );
}
