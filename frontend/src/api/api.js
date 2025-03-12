import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const healthCheck = async () => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    console.error('Health check failed:', error);
    throw error;
  }
};

export const getModelInfo = async () => {
  try {
    const response = await api.get('/model/info');
    return response.data;
  } catch (error) {
    console.error('Failed to get model info:', error);
    throw error;
  }
};

export const loadModel = async (config) => {
  try {
    const response = await api.post('/model/load', config);
    return response.data;
  } catch (error) {
    console.error('Failed to load model:', error);
    throw error;
  }
};

export const predict = async (features) => {
  try {
    const response = await api.post('/predict', { features });
    return response.data;
  } catch (error) {
    console.error('Prediction failed:', error);
    throw error;
  }
};

export const uploadData = async (file) => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await axios.post(`${API_URL}/upload/data`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return response.data;
  } catch (error) {
    console.error('File upload failed:', error);
    throw error;
  }
};

// Performance Metrics API Endpoints
export const getPerformanceMetrics = async (startDate, endDate, interval = 'daily') => {
  try {
    const params = {};
    if (startDate) params.start_date = startDate;
    if (endDate) params.end_date = endDate;
    if (interval) params.interval = interval;
    
    const response = await api.get('/api/metrics/performance', { params });
    return response.data;
  } catch (error) {
    console.error('Failed to get performance metrics:', error);
    throw error;
  }
};

export const getErrorLogs = async (startDate, endDate, errorType, limit = 100) => {
  try {
    const params = {};
    if (startDate) params.start_date = startDate;
    if (endDate) params.end_date = endDate;
    if (errorType) params.error_type = errorType;
    if (limit) params.limit = limit;
    
    const response = await api.get('/api/metrics/errors', { params });
    return response.data;
  } catch (error) {
    console.error('Failed to get error logs:', error);
    throw error;
  }
};

export const getMetricsSummary = async () => {
  try {
    const response = await api.get('/api/metrics/summary');
    return response.data;
  } catch (error) {
    console.error('Failed to get metrics summary:', error);
    throw error;
  }
};

export const getModelMetrics = async (modelId, metricName, startDate, endDate) => {
  try {
    const params = {};
    if (modelId) params.model_id = modelId;
    if (metricName) params.metric_name = metricName;
    if (startDate) params.start_date = startDate;
    if (endDate) params.end_date = endDate;
    
    const response = await api.get('/api/metrics/model', { params });
    return response.data;
  } catch (error) {
    console.error('Failed to get model metrics:', error);
    throw error;
  }
};

export default api; 