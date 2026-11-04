import React, { useState } from 'react';
import PredictionForm from '../components/PredictionForm';
import PredictionResult from '../components/PredictionResult';
import { predictTransaction } from '../services/api';

/**
 * Real-time Transaction Fraud Prediction Page.
 */
export default function Prediction({ apiStatus }) {
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handlePredict = async (transactionPayload) => {
    setIsLoading(true);
    setError('');
    setResult(null);

    try {
      const responseData = await predictTransaction(transactionPayload);
      setResult(responseData);
    } catch (err) {
      setError(err.message || 'An unexpected error occurred during prediction.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setError('');
  };

  return (
    <div>
      <div className="page-header">
        <h2 className="page-title">Real-Time Fraud Prediction</h2>
        <p className="page-description">
          Evaluate credit card transactions against the Scikit-learn Random Forest model via the FastAPI inference microservice.
        </p>
      </div>

      {apiStatus === 'disconnected' && (
        <div className="alert alert-warning">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
            <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>
            <line x1="12" x2="12" y1="9" y2="13"/>
            <line x1="12" x2="12.01" y1="17" y2="17"/>
          </svg>
          <div>
            <strong>Backend API is currently offline.</strong> Ensure the FastAPI server is running locally on <code>http://localhost:8000</code> before submitting transactions.
          </div>
        </div>
      )}

      <div className="prediction-layout">
        <div>
          <PredictionForm
            onSubmit={handlePredict}
            isLoading={isLoading}
            onReset={handleReset}
          />
        </div>

        <div>
          <PredictionResult
            result={result}
            isLoading={isLoading}
            error={error}
          />
        </div>
      </div>
    </div>
  );
}
