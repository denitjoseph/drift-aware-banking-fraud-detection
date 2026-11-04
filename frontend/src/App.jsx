import React, { useState, useEffect, useCallback } from 'react';
import Sidebar from './components/Sidebar';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import Prediction from './pages/Prediction';
import DatasetTraining from './pages/DatasetTraining';
import ModelInfo from './pages/ModelInfo';
import { checkHealth } from './services/api';

/**
 * Root Application Component.
 */
export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [apiStatus, setApiStatus] = useState('checking'); // 'connected' | 'disconnected' | 'checking'
  const [healthData, setHealthData] = useState(null);
  const [isCheckingHealth, setIsCheckingHealth] = useState(false);

  const fetchHealth = useCallback(async () => {
    setIsCheckingHealth(true);
    try {
      const data = await checkHealth();
      setHealthData(data);
      if (data && data.status === 'healthy') {
        setApiStatus('connected');
      } else {
        setApiStatus('disconnected');
      }
    } catch {
      setApiStatus('disconnected');
      setHealthData(null);
    } finally {
      setIsCheckingHealth(false);
    }
  }, []);

  useEffect(() => {
    fetchHealth();
    // Poll API status every 15 seconds
    const interval = setInterval(fetchHealth, 15000);
    return () => clearInterval(interval);
  }, [fetchHealth]);

  return (
    <div className="app-container">
      <Sidebar
        activeTab={activeTab}
        onSelectTab={setActiveTab}
      />

      <div className="main-wrapper">
        <Navbar
          apiStatus={apiStatus}
          healthData={healthData}
          onRefreshHealth={fetchHealth}
          isCheckingHealth={isCheckingHealth}
        />

        <main className="content-area">
          {activeTab === 'dashboard' && (
            <Dashboard
              apiStatus={apiStatus}
              healthData={healthData}
              onNavigateToPrediction={() => setActiveTab('prediction')}
              onNavigateToTraining={() => setActiveTab('dataset-training')}
            />
          )}

          {activeTab === 'prediction' && (
            <Prediction
              apiStatus={apiStatus}
            />
          )}

          {activeTab === 'dataset-training' && (
            <DatasetTraining
              apiStatus={apiStatus}
            />
          )}

          {activeTab === 'model-info' && (
            <ModelInfo
              healthData={healthData}
            />
          )}
        </main>
      </div>
    </div>
  );
}
