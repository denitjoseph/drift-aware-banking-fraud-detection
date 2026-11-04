import React from 'react';
import ApiStatus from './ApiStatus';

/**
 * Top Navbar component displaying platform branding and live API health.
 */
export default function Navbar({ apiStatus, healthData, onRefreshHealth, isCheckingHealth }) {
  return (
    <header className="top-navbar">
      <div className="navbar-title-wrap">
        <h1 className="navbar-title">Banking Fraud Detection Platform</h1>
        <p className="navbar-subtitle">AI-powered transaction monitoring and fraud prediction</p>
      </div>

      <div className="navbar-actions">
        <ApiStatus
          status={apiStatus}
          healthData={healthData}
          onRefresh={onRefreshHealth}
          isChecking={isCheckingHealth}
        />
      </div>
    </header>
  );
}
