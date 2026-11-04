import React, { useState } from 'react';

// Default initial state: all 30 features set to 0.0
export const DEFAULT_FORM_VALUES = {
  Time: 0.0,
  Amount: 0.0,
  ...Object.fromEntries(Array.from({ length: 28 }, (_, i) => [`V${i + 1}`, 0.0]))
};

// Known legitimate transaction from the Credit Card dataset (Class = 0)
export const SAMPLE_LEGITIMATE = {
  Time: 0.0,
  V1: -1.359807,
  V2: -0.072781,
  V3: 2.536347,
  V4: 1.378155,
  V5: -0.338321,
  V6: 0.462388,
  V7: 0.239599,
  V8: 0.098698,
  V9: 0.363787,
  V10: 0.090794,
  V11: -0.551600,
  V12: -0.617801,
  V13: -0.991390,
  V14: -0.311169,
  V15: 1.468177,
  V16: -0.470401,
  V17: 0.207971,
  V18: 0.025791,
  V19: 0.403993,
  V20: 0.251412,
  V21: -0.018307,
  V22: 0.277838,
  V23: -0.110474,
  V24: 0.066928,
  V25: 0.128539,
  V26: -0.189115,
  V27: 0.133558,
  V28: -0.021053,
  Amount: 149.62
};

// Known fraudulent transaction from the Credit Card dataset (Class = 1)
export const SAMPLE_FRAUD = {
  Time: 406.0,
  V1: -2.312227,
  V2: 1.951992,
  V3: -1.609851,
  V4: 3.997906,
  V5: -0.522188,
  V6: -1.426545,
  V7: -2.537387,
  V8: 1.391657,
  V9: -2.770089,
  V10: -2.772272,
  V11: 3.202033,
  V12: -2.899907,
  V13: -0.595222,
  V14: -4.289254,
  V15: 0.389724,
  V16: -1.140747,
  V17: -2.830056,
  V18: -0.016822,
  V19: 0.416956,
  V20: 0.126911,
  V21: 0.517232,
  V22: -0.035049,
  V23: -0.465211,
  V24: 0.320198,
  V25: 0.044519,
  V26: 0.177840,
  V27: 0.261145,
  V28: -0.143276,
  Amount: 0.0
};

/**
 * Transaction Prediction Form supporting manual input and real dataset presets.
 */
export default function PredictionForm({ onSubmit, isLoading, onReset }) {
  const [formData, setFormData] = useState(DEFAULT_FORM_VALUES);
  const [validationError, setValidationError] = useState('');

  const handleChange = (featureName, value) => {
    setFormData(prev => ({
      ...prev,
      [featureName]: value
    }));
  };

  const handleLoadSample = (sample) => {
    setFormData(sample);
    setValidationError('');
  };

  const handleReset = () => {
    setFormData(DEFAULT_FORM_VALUES);
    setValidationError('');
    if (onReset) onReset();
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    setValidationError('');

    // Parse and validate all 30 features as numeric
    const parsedPayload = {};
    for (const [key, val] of Object.entries(formData)) {
      const num = parseFloat(val);
      if (isNaN(num)) {
        setValidationError(`Feature "${key}" must be a valid number.`);
        return;
      }
      parsedPayload[key] = num;
    }

    if (parsedPayload.Amount < 0) {
      setValidationError('Amount must be a non-negative number (>= 0.0).');
      return;
    }

    onSubmit(parsedPayload);
  };

  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
        <h2 className="card-title" style={{ margin: 0 }}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
            <line x1="16" x2="8" y1="13" y2="13"/>
            <line x1="16" x2="8" y1="17" y2="17"/>
            <polyline points="10 9 9 9 8 9"/>
          </svg>
          Transaction Features (30 Features)
        </h2>
      </div>

      <p className="card-text">
        Input the 30 numerical features required by the Scikit-learn pipeline. Use the presets below to test known transactions from the dataset.
      </p>

      {/* Preset Action Bar */}
      <div className="preset-bar">
        <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-muted)', alignSelf: 'center' }}>
          Quick Load Presets:
        </span>
        <button
          type="button"
          className="btn-preset"
          onClick={() => handleLoadSample(SAMPLE_LEGITIMATE)}
          title="Load genuine transaction from dataset"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#16a34a" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="20 6 9 17 4 12"/>
          </svg>
          Sample Legitimate
        </button>

        <button
          type="button"
          className="btn-preset"
          onClick={() => handleLoadSample(SAMPLE_FRAUD)}
          title="Load confirmed fraudulent transaction from dataset"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#dc2626" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"/>
            <line x1="12" x2="12" y1="8" y2="12"/>
            <line x1="12" x2="12.01" y1="16" y2="16"/>
          </svg>
          Sample Fraud
        </button>

        <button
          type="button"
          className="btn-preset"
          onClick={handleReset}
          title="Reset all 30 features to default zero"
        >
          Reset to Zero
        </button>
      </div>

      {validationError && (
        <div className="alert alert-danger">
          <strong>Validation Error:</strong> {validationError}
        </div>
      )}

      <form onSubmit={handleSubmit}>
        {/* Top Primary Features: Time and Amount */}
        <div className="form-grid-top">
          <div className="form-group">
            <label className="form-label" htmlFor="input-time">
              Time (seconds elapsed):
            </label>
            <input
              id="input-time"
              type="number"
              step="any"
              className="form-input"
              value={formData.Time}
              onChange={(e) => handleChange('Time', e.target.value)}
              required
            />
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="input-amount">
              Amount (currency units &ge; 0):
            </label>
            <input
              id="input-amount"
              type="number"
              step="any"
              min="0"
              className="form-input"
              value={formData.Amount}
              onChange={(e) => handleChange('Amount', e.target.value)}
              required
            />
          </div>
        </div>

        {/* Compact Grid for PCA Features V1 - V28 */}
        <div className="form-section-title">
          PCA Features (V1 to V28)
        </div>

        <div className="form-grid-pca">
          {Array.from({ length: 28 }, (_, i) => {
            const featureName = `V${i + 1}`;
            return (
              <div key={featureName} className="form-group">
                <label className="form-label" htmlFor={`input-${featureName}`} style={{ fontSize: '11px' }}>
                  {featureName}:
                </label>
                <input
                  id={`input-${featureName}`}
                  type="number"
                  step="any"
                  className="form-input form-input-pca"
                  value={formData[featureName]}
                  onChange={(e) => handleChange(featureName, e.target.value)}
                  required
                />
              </div>
            );
          })}
        </div>

        {/* Action Buttons */}
        <div className="form-actions">
          <button
            type="submit"
            className="btn-primary"
            disabled={isLoading}
          >
            {isLoading ? (
              <>
                <span className="spinner"></span>
                Scoring Transaction...
              </>
            ) : (
              'Predict Transaction'
            )}
          </button>

          <button
            type="button"
            className="btn-secondary"
            onClick={handleReset}
            disabled={isLoading}
          >
            Reset Form
          </button>
        </div>
      </form>
    </div>
  );
}
