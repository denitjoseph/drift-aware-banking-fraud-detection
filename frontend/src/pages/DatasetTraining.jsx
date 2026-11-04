import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  uploadDataset,
  getDatasetPreview,
  listDatasets,
  startTraining,
  getTrainingStatus,
  listTrainings
} from '../services/api';

export default function DatasetTraining({ apiStatus }) {
  // Upload & Dataset State
  const [file, setFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState('');
  const [uploadedDataset, setUploadedDataset] = useState(null);
  const [previewData, setPreviewData] = useState(null);
  const [isLoadingPreview, setIsLoadingPreview] = useState(false);

  // Available datasets
  const [datasetsList, setDatasetsList] = useState([]);

  // Training State
  const [isStartingTraining, setIsStartingTraining] = useState(false);
  const [trainingError, setTrainingError] = useState('');
  const [activeTraining, setActiveTraining] = useState(null);
  const [trainingHistory, setTrainingHistory] = useState([]);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);

  const pollIntervalRef = useRef(null);
  const fileInputRef = useRef(null);

  // Fetch datasets list & training history on mount
  const refreshHistory = useCallback(async () => {
    setIsLoadingHistory(true);
    try {
      const [datasets, trainings] = await Promise.all([
        listDatasets().catch(() => []),
        listTrainings().catch(() => [])
      ]);
      setDatasetsList(datasets);
      setTrainingHistory(trainings);
    } catch {
      // Ignored if API is offline
    } finally {
      setIsLoadingHistory(false);
    }
  }, []);

  useEffect(() => {
    refreshHistory();
  }, [refreshHistory]);

  // Clean up polling timer on unmount
  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, []);

  // Poll active training until terminal state
  const pollTraining = useCallback((trainingId) => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
    }

    pollIntervalRef.current = setInterval(async () => {
      try {
        const record = await getTrainingStatus(trainingId);
        setActiveTraining(record);

        if (record.status === 'completed' || record.status === 'failed') {
          clearInterval(pollIntervalRef.current);
          pollIntervalRef.current = null;
          refreshHistory();
        }
      } catch (err) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
        setTrainingError(err.message || 'Failed to poll training status');
      }
    }, 1500);
  }, [refreshHistory]);

  // Handle file select
  const handleFileChange = (e) => {
    const selected = e.target.files?.[0];
    setUploadError('');
    if (!selected) {
      setFile(null);
      return;
    }
    if (!selected.name.toLowerCase().endsWith('.csv')) {
      setUploadError('Only CSV (.csv) files are supported.');
      setFile(null);
      return;
    }
    setFile(selected);
  };

  // Upload handler
  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) {
      setUploadError('Please select a CSV dataset file first.');
      return;
    }

    setIsUploading(true);
    setUploadError('');

    try {
      const result = await uploadDataset(file);
      setUploadedDataset(result);
      await loadPreview(result.dataset_id);
      await refreshHistory();
      setFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    } catch (err) {
      setUploadError(err.message || 'Failed to upload dataset.');
    } finally {
      setIsUploading(false);
    }
  };

  // Load preview rows
  const loadPreview = async (datasetId) => {
    setIsLoadingPreview(true);
    try {
      const data = await getDatasetPreview(datasetId, 5);
      setPreviewData(data);
    } catch (err) {
      setUploadError(`Failed to load dataset preview: ${err.message}`);
    } finally {
      setIsLoadingPreview(false);
    }
  };

  // Select an existing dataset from list
  const handleSelectExistingDataset = async (dataset) => {
    setUploadedDataset(dataset);
    setUploadError('');
    await loadPreview(dataset.dataset_id);
  };

  // Start training handler
  const handleStartTraining = async () => {
    const datasetId = uploadedDataset?.dataset_id;
    if (!datasetId) {
      setTrainingError('No dataset selected for training.');
      return;
    }

    setIsStartingTraining(true);
    setTrainingError('');

    try {
      const record = await startTraining(datasetId);
      setActiveTraining(record);
      pollTraining(record.training_id);
    } catch (err) {
      setTrainingError(err.message || 'Failed to start model training.');
    } finally {
      setIsStartingTraining(false);
    }
  };

  // Select an old training run to inspect its metrics
  const handleSelectTrainingRun = (run) => {
    setActiveTraining(run);
  };

  // Format bytes
  const formatBytes = (bytes) => {
    if (!bytes && bytes !== 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const isConnected = apiStatus === 'connected';
  const validation = uploadedDataset?.validation || previewData?.validation;
  const isValidDataset = Boolean(validation?.is_valid);
  const isTrainingActive = activeTraining?.status === 'queued' || activeTraining?.status === 'running';

  return (
    <div className="page-container">
      {/* Page Header */}
      <div className="page-header">
        <div>
          <h1 className="page-title">Dataset Upload & Model Training</h1>
          <p className="page-subtitle">
            Upload custom CSV transaction data, validate credit card fraud features, and trigger asynchronous Scikit-learn Random Forest model retraining with holdout evaluation.
          </p>
        </div>
        <button
          className="btn btn-secondary"
          onClick={refreshHistory}
          disabled={isLoadingHistory || !isConnected}
          title="Refresh history and datasets"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>
            <path d="M3 3v5h5"/>
            <path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16"/>
            <path d="M21 21v-5h-5"/>
          </svg>
          Refresh Data
        </button>
      </div>

      {!isConnected && (
        <div className="alert alert-warning" style={{ marginBottom: '24px' }}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10"/>
            <line x1="12" y1="8" x2="12" y2="12"/>
            <line x1="12" y1="16" x2="12.01" y2="16"/>
          </svg>
          <div>
            <strong>Backend API is currently disconnected.</strong> Uploading and model retraining require the FastAPI server running on port 8000.
          </div>
        </div>
      )}

      {/* Grid: Upload & Dataset Info */}
      <div className="dashboard-grid" style={{ marginBottom: '24px' }}>
        {/* Upload Form Card */}
        <div className="card">
          <div className="card-header">
            <div>
              <h2 className="card-title">1. Upload CSV Dataset</h2>
              <p className="card-subtitle">Accepts credit card transactions CSV with 30 features + Class target</p>
            </div>
            <span className="badge badge-primary">Supervised ML</span>
          </div>

          <form onSubmit={handleUpload}>
            <div className="file-dropzone">
              <input
                ref={fileInputRef}
                type="file"
                id="dataset-file-input"
                accept=".csv"
                onChange={handleFileChange}
                disabled={isUploading || !isConnected}
                style={{ display: 'none' }}
              />
              <label htmlFor="dataset-file-input" className="file-dropzone-label">
                <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#2563eb" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                  <polyline points="17 8 12 3 7 8"/>
                  <line x1="12" x2="12" y1="3" y2="15"/>
                </svg>
                <div style={{ marginTop: '10px', fontWeight: 600, color: 'var(--text-primary)' }}>
                  {file ? file.name : 'Click to select CSV dataset'}
                </div>
                <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px' }}>
                  {file ? `${formatBytes(file.size)} selected` : 'CSV format up to 100 MB'}
                </div>
              </label>
            </div>

            {uploadError && (
              <div className="alert alert-danger" style={{ marginTop: '14px' }}>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>
                </svg>
                <span>{uploadError}</span>
              </div>
            )}

            <div style={{ display: 'flex', gap: '12px', marginTop: '16px' }}>
              <button
                type="submit"
                className="btn btn-primary"
                disabled={!file || isUploading || !isConnected}
                style={{ flex: 1 }}
              >
                {isUploading ? (
                  <>
                    <span className="spinner-border" />
                    Uploading & Validating...
                  </>
                ) : (
                  <>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                      <polyline points="17 8 12 3 7 8"/>
                      <line x1="12" x2="12" y1="3" y2="15"/>
                    </svg>
                    Upload & Validate
                  </>
                )}
              </button>

              {file && (
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => {
                    setFile(null);
                    setUploadError('');
                    if (fileInputRef.current) fileInputRef.current.value = '';
                  }}
                  disabled={isUploading}
                >
                  Clear
                </button>
              )}
            </div>
          </form>

          {/* Quick dataset selector if uploaded before */}
          {datasetsList.length > 0 && (
            <div style={{ marginTop: '20px', paddingTop: '16px', borderTop: '1px solid var(--border-color)' }}>
              <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '8px' }}>
                Or select previously uploaded dataset:
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                {datasetsList.slice(0, 5).map((d) => (
                  <button
                    key={d.dataset_id}
                    type="button"
                    className={`btn-chip ${uploadedDataset?.dataset_id === d.dataset_id ? 'active' : ''}`}
                    onClick={() => handleSelectExistingDataset(d)}
                  >
                    📄 {d.filename} ({d.row_count} rows)
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Dataset Schema Validation Card */}
        <div className="card">
          <div className="card-header">
            <div>
              <h2 className="card-title">Dataset Schema & Class Distribution</h2>
              <p className="card-subtitle">Automated validation for supervised model training</p>
            </div>
            {uploadedDataset && (
              <span className={`badge ${isValidDataset ? 'badge-success' : 'badge-danger'}`}>
                {isValidDataset ? 'Validation Passed' : 'Validation Failed'}
              </span>
            )}
          </div>

          {!uploadedDataset ? (
            <div className="empty-state-box">
              <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" strokeWidth="1.5">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                <polyline points="14 2 14 8 20 8"/>
                <line x1="16" y1="13" x2="8" y2="13"/>
                <line x1="16" y1="17" x2="8" y2="17"/>
              </svg>
              <p style={{ marginTop: '10px', color: 'var(--text-muted)' }}>
                Upload or select a dataset to inspect its columns, missing values, and class balance.
              </p>
            </div>
          ) : (
            <div>
              <div className="stat-grid" style={{ marginBottom: '16px' }}>
                <div className="mini-stat">
                  <div className="mini-stat-label">Dataset ID</div>
                  <div className="mini-stat-value" style={{ fontSize: '13px', fontFamily: 'monospace' }}>
                    {uploadedDataset.dataset_id}
                  </div>
                </div>
                <div className="mini-stat">
                  <div className="mini-stat-label">File Size</div>
                  <div className="mini-stat-value">{formatBytes(uploadedDataset.file_size_bytes)}</div>
                </div>
                <div className="mini-stat">
                  <div className="mini-stat-label">Total Rows</div>
                  <div className="mini-stat-value">{uploadedDataset.row_count?.toLocaleString()}</div>
                </div>
                <div className="mini-stat">
                  <div className="mini-stat-label">Total Columns</div>
                  <div className="mini-stat-value">{uploadedDataset.column_count}</div>
                </div>
              </div>

              {/* Class distribution */}
              {validation?.class_distribution && (
                <div className="distribution-box" style={{ marginBottom: '16px' }}>
                  <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '8px' }}>
                    Class Target Balance:
                  </div>
                  <div style={{ display: 'flex', gap: '12px' }}>
                    <div className="dist-pill dist-legit" style={{ flex: 1 }}>
                      <span className="dist-pill-dot dot-green" />
                      <span>Legitimate (Class 0):</span>
                      <strong>{validation.class_distribution.legitimate?.toLocaleString()}</strong>
                    </div>
                    <div className="dist-pill dist-fraud" style={{ flex: 1 }}>
                      <span className="dist-pill-dot dot-red" />
                      <span>Fraud (Class 1):</span>
                      <strong>{validation.class_distribution.fraud?.toLocaleString()}</strong>
                    </div>
                  </div>
                </div>
              )}

              {/* Validation errors */}
              {validation?.errors && validation.errors.length > 0 && (
                <div className="alert alert-danger" style={{ marginBottom: '12px' }}>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
                  </svg>
                  <div>
                    <strong>Validation Blockers ({validation.errors.length}):</strong>
                    <ul style={{ paddingLeft: '18px', marginTop: '4px' }}>
                      {validation.errors.map((err, i) => (
                        <li key={i}>{err}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}

              {/* Validation warnings */}
              {validation?.warnings && validation.warnings.length > 0 && (
                <div className="alert alert-warning">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
                  </svg>
                  <div>
                    <strong>Notes / Warnings ({validation.warnings.length}):</strong>
                    <ul style={{ paddingLeft: '18px', marginTop: '4px' }}>
                      {validation.warnings.map((w, i) => (
                        <li key={i}>{w}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Dataset Preview Section */}
      {previewData && (
        <div className="card" style={{ marginBottom: '24px' }}>
          <div className="card-header">
            <div>
              <h2 className="card-title">Dataset Rows Preview</h2>
              <p className="card-subtitle">Showing first {previewData.sample_rows?.length || 0} sample records</p>
            </div>
            <span className="badge badge-neutral">{previewData.columns?.length || 0} Features</span>
          </div>

          {isLoadingPreview ? (
            <div style={{ textAlign: 'center', padding: '30px', color: 'var(--text-muted)' }}>
              <span className="spinner-border" /> Loading sample records...
            </div>
          ) : (
            <div className="table-wrapper" style={{ maxHeight: '260px', overflow: 'auto' }}>
              <table className="preview-table">
                <thead>
                  <tr>
                    {previewData.columns?.map((col) => (
                      <th
                        key={col}
                        className={col === 'Class' ? 'th-target' : col === 'Time' || col === 'Amount' ? 'th-highlight' : ''}
                      >
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {previewData.sample_rows?.map((row, idx) => (
                    <tr key={idx}>
                      {previewData.columns?.map((col) => (
                        <td
                          key={col}
                          className={col === 'Class' ? (row[col] === 1 ? 'td-fraud' : 'td-legit') : ''}
                        >
                          {typeof row[col] === 'number'
                            ? Number.isInteger(row[col])
                              ? row[col]
                              : row[col].toFixed(4)
                            : String(row[col] ?? '')}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* Model Retraining Execution Card */}
      <div className="card" style={{ marginBottom: '24px' }}>
        <div className="card-header">
          <div>
            <h2 className="card-title">2. Train Random Forest Model</h2>
            <p className="card-subtitle">
              Trains ColumnTransformer + RandomForestClassifier with balanced class weights on 80/20 train/holdout split.
            </p>
          </div>
          {activeTraining && (
            <span
              className={`badge ${
                activeTraining.status === 'completed'
                  ? 'badge-success'
                  : activeTraining.status === 'running'
                  ? 'badge-warning badge-pulse'
                  : activeTraining.status === 'failed'
                  ? 'badge-danger'
                  : 'badge-primary'
              }`}
            >
              Status: {activeTraining.status.toUpperCase()}
            </span>
          )}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            <p style={{ color: 'var(--text-secondary)', fontSize: '14px', maxWidth: '680px', margin: 0 }}>
              The training pipeline fits a Standard Scaler on <code>Time</code> & <code>Amount</code>, trains balanced decision trees, applies the <code>0.40</code> classification cutoff, and persists a candidate artifact in <code>backend/models/trained/</code> without modifying the live production model.
            </p>
          </div>

          <button
            type="button"
            className="btn btn-primary btn-lg"
            onClick={handleStartTraining}
            disabled={!isValidDataset || isTrainingActive || isStartingTraining || !isConnected}
          >
            {isTrainingActive || isStartingTraining ? (
              <>
                <span className="spinner-border" />
                Training in Background...
              </>
            ) : (
              <>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polygon points="5 3 19 12 5 21 5 3"/>
                </svg>
                Start Model Training
              </>
            )}
          </button>
        </div>

        {trainingError && (
          <div className="alert alert-danger" style={{ marginTop: '16px' }}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
            </svg>
            <span>{trainingError}</span>
          </div>
        )}

        {/* Active training details */}
        {activeTraining && (
          <div className="training-status-box" style={{ marginTop: '20px' }}>
            <div className="training-status-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                {isTrainingActive && <span className="spinner-border" style={{ borderColor: 'var(--color-primary)' }} />}
                <span style={{ fontWeight: 600 }}>Job ID: <code>{activeTraining.training_id}</code></span>
                <span style={{ color: 'var(--text-muted)', fontSize: '13px' }}>
                  (Dataset: {activeTraining.dataset_id})
                </span>
              </div>
              <div style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
                Started: {new Date(activeTraining.created_at).toLocaleTimeString()}
                {activeTraining.completed_at && ` • Finished: ${new Date(activeTraining.completed_at).toLocaleTimeString()}`}
              </div>
            </div>

            {activeTraining.status === 'failed' && (
              <div className="alert alert-danger" style={{ marginTop: '12px' }}>
                <strong>Training Error:</strong> {activeTraining.error || 'Unknown training error occurred.'}
              </div>
            )}

            {/* Candidate Model Metrics Display */}
            {activeTraining.status === 'completed' && activeTraining.metrics && (
              <div style={{ marginTop: '20px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
                  <h3 style={{ fontSize: '15px', fontWeight: 700, color: 'var(--text-primary)' }}>
                    Holdout Evaluation Metrics (Test Split @ Cutoff 0.40)
                  </h3>
                  <span className="badge badge-success">Candidate Model Saved</span>
                </div>

                <div className="metrics-grid">
                  <div className="metric-card">
                    <div className="metric-label">Precision</div>
                    <div className="metric-value">
                      {(activeTraining.metrics.precision * 100).toFixed(1)}%
                    </div>
                    <div className="metric-sub">True Positives / Predicted Positives</div>
                  </div>

                  <div className="metric-card">
                    <div className="metric-label">Recall</div>
                    <div className="metric-value">
                      {(activeTraining.metrics.recall * 100).toFixed(1)}%
                    </div>
                    <div className="metric-sub">True Positives / Actual Fraud</div>
                  </div>

                  <div className="metric-card">
                    <div className="metric-label">F1-Score</div>
                    <div className="metric-value">
                      {activeTraining.metrics.f1_score.toFixed(4)}
                    </div>
                    <div className="metric-sub">Harmonic mean of Precision & Recall</div>
                  </div>

                  <div className="metric-card">
                    <div className="metric-label">PR-AUC</div>
                    <div className="metric-value">
                      {activeTraining.metrics.pr_auc.toFixed(4)}
                    </div>
                    <div className="metric-sub">Area under Precision-Recall Curve</div>
                  </div>
                </div>

                {/* Confusion Matrix & Split details */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px', marginTop: '16px' }}>
                  {/* Confusion Matrix */}
                  <div className="confusion-matrix-box">
                    <div style={{ fontWeight: 600, fontSize: '13px', marginBottom: '10px', color: 'var(--text-secondary)' }}>
                      Holdout Confusion Matrix:
                    </div>
                    <div className="cm-grid">
                      <div className="cm-cell cm-tn">
                        <div className="cm-val">{activeTraining.metrics.confusion_matrix.true_negatives}</div>
                        <div className="cm-lbl">True Negatives (TN)</div>
                      </div>
                      <div className="cm-cell cm-fp">
                        <div className="cm-val">{activeTraining.metrics.confusion_matrix.false_positives}</div>
                        <div className="cm-lbl">False Positives (FP)</div>
                      </div>
                      <div className="cm-cell cm-fn">
                        <div className="cm-val">{activeTraining.metrics.confusion_matrix.false_negatives}</div>
                        <div className="cm-lbl">False Negatives (FN)</div>
                      </div>
                      <div className="cm-cell cm-tp">
                        <div className="cm-val">{activeTraining.metrics.confusion_matrix.true_positives}</div>
                        <div className="cm-lbl">True Positives (TP)</div>
                      </div>
                    </div>
                  </div>

                  {/* Split Summary */}
                  <div className="split-summary-box">
                    <div style={{ fontWeight: 600, fontSize: '13px', marginBottom: '10px', color: 'var(--text-secondary)' }}>
                      Dataset Split Details:
                    </div>
                    <ul className="split-list">
                      <li>
                        <span>Total Samples:</span>
                        <strong>{activeTraining.metrics.total_samples?.toLocaleString()}</strong>
                      </li>
                      <li>
                        <span>Train Set (80%):</span>
                        <strong>{activeTraining.metrics.train_samples?.toLocaleString()}</strong>
                      </li>
                      <li>
                        <span>Holdout Test Set (20%):</span>
                        <strong>{activeTraining.metrics.test_samples?.toLocaleString()}</strong>
                      </li>
                      <li>
                        <span>Decision Cutoff:</span>
                        <strong>{activeTraining.metrics.threshold}</strong>
                      </li>
                      <li>
                        <span>Model Artifact:</span>
                        <code style={{ fontSize: '11px' }}>{activeTraining.training_id}_pipeline.pkl</code>
                      </li>
                    </ul>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Training History Table */}
      <div className="card">
        <div className="card-header">
          <div>
            <h2 className="card-title">Training History</h2>
            <p className="card-subtitle">Previous model training sessions and candidate checkpoints</p>
          </div>
          <span className="badge badge-neutral">{trainingHistory.length} Runs</span>
        </div>

        {trainingHistory.length === 0 ? (
          <div className="empty-state-box">
            <p style={{ color: 'var(--text-muted)' }}>No model training runs recorded yet. Upload a dataset and train your first candidate model!</p>
          </div>
        ) : (
          <div className="table-wrapper">
            <table className="history-table">
              <thead>
                <tr>
                  <th>Job ID</th>
                  <th>Dataset ID</th>
                  <th>Status</th>
                  <th>F1-Score</th>
                  <th>PR-AUC</th>
                  <th>Created At</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {trainingHistory.map((run) => (
                  <tr
                    key={run.training_id}
                    className={activeTraining?.training_id === run.training_id ? 'tr-selected' : ''}
                  >
                    <td><code>{run.training_id}</code></td>
                    <td><code>{run.dataset_id}</code></td>
                    <td>
                      <span
                        className={`badge ${
                          run.status === 'completed'
                            ? 'badge-success'
                            : run.status === 'running'
                            ? 'badge-warning'
                            : run.status === 'failed'
                            ? 'badge-danger'
                            : 'badge-primary'
                        }`}
                      >
                        {run.status}
                      </span>
                    </td>
                    <td>{run.metrics ? run.metrics.f1_score.toFixed(4) : '—'}</td>
                    <td>{run.metrics ? run.metrics.pr_auc.toFixed(4) : '—'}</td>
                    <td>{new Date(run.created_at).toLocaleString()}</td>
                    <td>
                      <button
                        type="button"
                        className="btn-chip"
                        onClick={() => handleSelectTrainingRun(run)}
                      >
                        Inspect
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

