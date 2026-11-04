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

export { API_BASE_URL };
