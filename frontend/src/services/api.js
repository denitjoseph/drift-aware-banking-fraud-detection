/**
 * API Service for communication with the FastAPI Banking Fraud Detection backend.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

/**
 * Check root endpoint availability.
 */
export async function checkRoot() {
  try {
    const response = await fetch(`${API_BASE_URL}/`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' }
    });
    if (!response.ok) {
      throw new Error(`Server returned HTTP ${response.status}`);
    }
    return await response.json();
  } catch (err) {
    throw new Error(err.message || 'Failed to reach API root');
  }
}

/**
 * Check backend health status and model readiness.
 */
export async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' }
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(
        errorData.detail || `Health check failed with status ${response.status}`
      );
    }

    return await response.json();
  } catch (err) {
    // Network / connection error
    if (err.name === 'TypeError' && err.message.includes('fetch')) {
      throw new Error(`Unable to connect to backend at ${API_BASE_URL}. Ensure FastAPI is running.`);
    }
    throw err;
  }
}

/**
 * Submit transaction features to the fraud prediction endpoint.
 * @param {Object} transaction - 30 transaction features (Time, V1..V28, Amount)
 * @returns {Promise<Object>} { fraud_probability, classification_threshold, prediction }
 */
export async function predictTransaction(transaction) {
  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify(transaction)
    });

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      // 422 Validation error
      if (response.status === 422 && Array.isArray(data.detail)) {
        const errorMessages = data.detail
          .map(item => `${item.loc?.slice(-1)[0] || 'field'}: ${item.msg}`)
          .join(', ');
        throw new Error(`Validation Error: ${errorMessages}`);
      }

      // 503 Model not ready
      if (response.status === 503) {
        throw new Error(data.detail || 'Model service is unavailable. Model pipeline may not be loaded.');
      }

      // Generic error with backend detail
      throw new Error(data.detail || `Inference request failed with HTTP ${response.status}`);
    }

    return data;
  } catch (err) {
    if (err.name === 'TypeError' && err.message.includes('fetch')) {
      throw new Error(`Unable to reach backend at ${API_BASE_URL}. Check if the server is running on port 8000.`);
    }
    throw err;
  }
}

/**
 * Upload a CSV dataset for fraud detection model training.
 * @param {File} file - CSV file from file input
 * @returns {Promise<Object>} DatasetUploadResponse
 */
export async function uploadDataset(file) {
  try {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE_URL}/datasets/upload`, {
      method: 'POST',
      body: formData
    });

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      throw new Error(data.detail || `Upload failed with HTTP ${response.status}`);
    }

    return data;
  } catch (err) {
    if (err.name === 'TypeError' && err.message.includes('fetch')) {
      throw new Error(`Unable to reach backend at ${API_BASE_URL}. Ensure FastAPI is running.`);
    }
    throw err;
  }
}

/**
 * Fetch preview rows and validation status for an uploaded dataset.
 * @param {string} datasetId
 * @param {number} limit
 */
export async function getDatasetPreview(datasetId, limit = 5) {
  try {
    const response = await fetch(`${API_BASE_URL}/datasets/${datasetId}/preview?limit=${limit}`);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.detail || `Failed to fetch dataset preview: ${response.status}`);
    }
    return data;
  } catch (err) {
    throw err;
  }
}

/**
 * List all uploaded datasets.
 */
export async function listDatasets() {
  try {
    const response = await fetch(`${API_BASE_URL}/datasets`);
    const data = await response.json().catch(() => ([]));
    if (!response.ok) {
      throw new Error(`Failed to list datasets: ${response.status}`);
    }
    return Array.isArray(data) ? data : [];
  } catch (err) {
    throw err;
  }
}

/**
 * Trigger asynchronous model training on an uploaded dataset.
 * @param {string} datasetId
 * @returns {Promise<Object>} TrainingRecord
 */
export async function startTraining(datasetId) {
  try {
    const response = await fetch(`${API_BASE_URL}/training`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify({ dataset_id: datasetId })
    });

    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.detail || `Failed to start training: ${response.status}`);
    }
    return data;
  } catch (err) {
    if (err.name === 'TypeError' && err.message.includes('fetch')) {
      throw new Error(`Unable to reach backend at ${API_BASE_URL}. Ensure FastAPI is running.`);
    }
    throw err;
  }
}

/**
 * Fetch training job status and metrics.
 * @param {string} trainingId
 * @returns {Promise<Object>} TrainingRecord
 */
export async function getTrainingStatus(trainingId) {
  try {
    const response = await fetch(`${API_BASE_URL}/training/${trainingId}`);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.detail || `Failed to fetch training status: ${response.status}`);
    }
    return data;
  } catch (err) {
    throw err;
  }
}

/**
 * List all training sessions.
 * @returns {Promise<Array>} List of TrainingRecord
 */
export async function listTrainings() {
  try {
    const response = await fetch(`${API_BASE_URL}/training`);
    const data = await response.json().catch(() => ([]));
    if (!response.ok) {
      throw new Error(`Failed to list training runs: ${response.status}`);
    }
    return Array.isArray(data) ? data : [];
  } catch (err) {
    throw err;
  }
}

export { API_BASE_URL };
