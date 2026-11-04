import React from 'react';

/**
 * Model Information Page documenting pipeline components and configuration parameters.
 */
export default function ModelInfo({ healthData }) {
  const modelName = healthData?.model_name || 'Random Forest';
  const threshold = healthData?.classification_threshold || 0.40;

  return (
    <div>
      <div className="page-header">
        <h2 className="page-title">Model Architecture & Specifications</h2>
        <p className="page-description">
          Verified machine learning specifications from Phase 2 development and Phase 3 FastAPI serving configuration.
        </p>
      </div>

      <div className="info-grid">
        <div className="card">
          <h3 className="card-title">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect width="18" height="18" x="3" y="3" rx="2"/>
              <path d="M3 9h18"/>
              <path d="M9 21V9"/>
            </svg>
            Production Pipeline Summary
          </h3>

          <p className="card-text">
            The platform utilizes a unified <strong>Scikit-learn Pipeline</strong> persisted at <code>models/fraud_detection_pipeline.pkl</code>. Preprocessing and classifier inference are bundled together to ensure consistent scaling transformations on unscaled raw inputs.
          </p>

          <table className="data-table">
            <tbody>
              <tr>
                <td style={{ fontWeight: 600, width: '220px' }}>Classifier Algorithm</td>
                <td>{modelName} (100 estimators, balanced class weights, random_state=42)</td>
              </tr>
              <tr>
                <td style={{ fontWeight: 600 }}>Feature Transformation</td>
                <td>StandardScaler applied to <code>Time</code> and <code>Amount</code>; <code>V1</code>–<code>V28</code> passthrough</td>
              </tr>
              <tr>
                <td style={{ fontWeight: 600 }}>Decision Threshold</td>
                <td><strong>{threshold}</strong> (selected via validation set to maximize F1-score)</td>
              </tr>
              <tr>
                <td style={{ fontWeight: 600 }}>Source Dataset</td>
                <td>Kaggle Credit Card Fraud Detection (284,807 raw transactions, 283,726 deduplicated)</td>
              </tr>
              <tr>
                <td style={{ fontWeight: 600 }}>Target Distribution</td>
                <td>0.17% Fraudulent (Class 1) vs 99.83% Legitimate (Class 0)</td>
              </tr>
              <tr>
                <td style={{ fontWeight: 600 }}>Current Deployment Mode</td>
                <td>Local Development (FastAPI + Uvicorn) &amp; Docker container (Port 8000)</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="card">
          <h3 className="card-title">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            </svg>
            MLOps Architecture Boundary
          </h3>
          <p className="card-text" style={{ fontSize: '13px' }}>
            This project demonstrates a real-world MLOps implementation pathway.
          </p>
          <div className="alert alert-info" style={{ fontSize: '12px', marginBottom: '12px' }}>
            <strong>Local Dev Stage:</strong> The model runs inside the local Python virtual environment or Docker desktop container. Cloud deployment (Kubernetes on AWS EKS) and CI/CD pipelines will be configured in subsequent DevOps phases.
          </div>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            <strong>Config File:</strong> <code>configs/model_config.json</code><br/>
            <strong>Pipeline Artifact:</strong> <code>models/fraud_detection_pipeline.pkl</code>
          </div>
        </div>
      </div>
    </div>
  );
}
