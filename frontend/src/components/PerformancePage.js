import React, { useState, useEffect } from 'react';
import { 
  Container, Typography, Paper, Box, Grid, Card, CardContent, 
  CircularProgress, Alert, Tabs, Tab, Table, TableBody, TableCell, 
  TableContainer, TableHead, TableRow, Divider, Chip, IconButton,
  MenuItem, Select, FormControl, InputLabel, Tooltip
} from '@mui/material';
import { 
  Timeline, Speed, Error as ErrorIcon, CheckCircle, 
  TrendingUp, TrendingDown, TrendingFlat, Refresh
} from '@mui/icons-material';
import { 
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, 
  Tooltip as RechartsTooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell
} from 'recharts';
import { getPerformanceMetrics, getModelInfo, getErrorLogs, getMetricsSummary } from '../api/api';

// Colors for charts
const COLORS = ['#3f51b5', '#f50057', '#00bcd4', '#ff9800', '#4caf50', '#9c27b0'];
const STATUS_COLORS = {
  '2xx': '#4caf50',
  '3xx': '#03a9f4',
  '4xx': '#ff9800',
  '5xx': '#f44336'
};

// Helper function to generate mock data for development
const generateMockData = () => {
  const summary = {
    current: {
      total_requests: 1250,
      error_rate: 2.4,
      avg_response_time: 0.125
    },
    previous: {
      total_requests: 1100,
      error_rate: 3.1,
      avg_response_time: 0.142
    },
    trends: {
      requests: { percentage: 13.6, direction: 'up' },
      error_rate: { percentage: 22.6, direction: 'up' },
      response_time: { percentage: 12.0, direction: 'up' }
    },
    top_endpoints: [
      { endpoint: '/api/model/info', count: 450, avg_response_time: 0.085 },
      { endpoint: '/api/predict', count: 320, avg_response_time: 0.215 },
      { endpoint: '/api/metrics/summary', count: 180, avg_response_time: 0.095 },
      { endpoint: '/api/model/load', count: 120, avg_response_time: 0.320 },
      { endpoint: '/api/upload/data', count: 80, avg_response_time: 0.450 }
    ],
    recent_errors: [
      { timestamp: '2025-03-11T20:45:12', endpoint: '/api/predict', status_code: 400, error: 'Invalid input format' },
      { timestamp: '2025-03-11T19:32:05', endpoint: '/api/model/load', status_code: 500, error: 'Failed to load model: File not found' },
      { timestamp: '2025-03-11T18:15:47', endpoint: '/api/upload/data', status_code: 413, error: 'File too large' }
    ]
  };

  // Generate time series data
  const timeSeriesData = {};
  const now = new Date();
  for (let i = 6; i >= 0; i--) {
    const date = new Date(now);
    date.setDate(date.getDate() - i);
    const dateStr = date.toISOString().split('T')[0];
    
    timeSeriesData[dateStr] = [
      { status_code: 200, count: Math.floor(Math.random() * 500) + 500, avg_response_time: Math.random() * 0.1 + 0.05 },
      { status_code: 400, count: Math.floor(Math.random() * 20), avg_response_time: Math.random() * 0.2 + 0.1 },
      { status_code: 500, count: Math.floor(Math.random() * 10), avg_response_time: Math.random() * 0.3 + 0.2 }
    ];
  }

  return {
    summary,
    time_series: timeSeriesData,
    endpoints: summary.top_endpoints
  };
};

const PerformancePage = () => {
  const [metrics, setMetrics] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [tabValue, setTabValue] = useState(0);
  const [timeRange, setTimeRange] = useState('week');
  const [chartData, setChartData] = useState([]);
  const [statusDistribution, setStatusDistribution] = useState([]);

  useEffect(() => {
    fetchData();
  }, [timeRange]);

  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);

      // In a production environment, you would use these API calls
      // const metricsData = await getPerformanceMetrics();
      // const modelData = await getModelInfo();
      // const summaryData = await getMetricsSummary();

      // For development, use mock data
      const mockData = generateMockData();
      setMetrics(mockData);
      
      // Mock model info
      setModelInfo({
        loaded: true,
        input_size: 20,
        hidden_size: 64,
        output_size: 2,
        optimized: true,
        cached: true
      });

      // Process chart data
      processChartData(mockData);
    } catch (err) {
      setError('Failed to fetch performance metrics. Please try again later.');
      console.error('Error fetching metrics:', err);
    } finally {
      setLoading(false);
    }
  };

  const processChartData = (data) => {
    if (!data || !data.time_series) return;

    // Process time series data for charts
    const chartData = Object.entries(data.time_series).map(([date, statuses]) => {
      const entry = { date };
      let total = 0;
      
      statuses.forEach(status => {
        const statusGroup = Math.floor(status.status_code / 100) + 'xx';
        entry[statusGroup] = status.count;
        entry[`${statusGroup}_time`] = status.avg_response_time;
        total += status.count;
      });
      
      entry.total = total;
      return entry;
    });

    setChartData(chartData);

    // Process status distribution for pie chart
    const statusCounts = {};
    Object.values(data.time_series).forEach(statuses => {
      statuses.forEach(status => {
        const statusGroup = Math.floor(status.status_code / 100) + 'xx';
        statusCounts[statusGroup] = (statusCounts[statusGroup] || 0) + status.count;
      });
    });

    const statusDistribution = Object.entries(statusCounts).map(([name, value]) => ({
      name,
      value
    }));

    setStatusDistribution(statusDistribution);
  };

  const calculateSummary = () => {
    if (!metrics || !metrics.summary) return null;

    const { current, previous, trends } = metrics.summary;

    return (
      <Grid container spacing={3}>
        {/* Request Volume */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6" color="text.secondary">
                  Request Volume
                </Typography>
                <Timeline color="primary" />
              </Box>
              
              <Typography variant="h4">
                {current.total_requests.toLocaleString()}
              </Typography>
              
              <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                {trends.requests.direction === 'up' ? (
                  <TrendingUp color="success" />
                ) : trends.requests.direction === 'down' ? (
                  <TrendingDown color="error" />
                ) : (
                  <TrendingFlat color="action" />
                )}
                <Typography variant="body2" color="text.secondary" sx={{ ml: 1 }}>
                  {trends.requests.percentage.toFixed(1)}% {trends.requests.direction} from yesterday
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        {/* Response Time */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6" color="text.secondary">
                  Avg Response Time
                </Typography>
                <Speed color="primary" />
              </Box>
              
              <Typography variant="h4">
                {(current.avg_response_time * 1000).toFixed(2)} ms
              </Typography>
              
              <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                {trends.response_time.direction === 'up' ? (
                  <TrendingUp color={trends.response_time.direction === 'up' ? 'error' : 'success'} />
                ) : trends.response_time.direction === 'down' ? (
                  <TrendingDown color={trends.response_time.direction === 'down' ? 'success' : 'error'} />
                ) : (
                  <TrendingFlat color="action" />
                )}
                <Typography variant="body2" color="text.secondary" sx={{ ml: 1 }}>
                  {trends.response_time.percentage.toFixed(1)}% {trends.response_time.direction} from yesterday
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        {/* Error Rate */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6" color="text.secondary">
                  Error Rate
                </Typography>
                <ErrorIcon color="primary" />
              </Box>
              
              <Typography variant="h4">
                {current.error_rate.toFixed(2)}%
              </Typography>
              
              <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                {trends.error_rate.direction === 'up' ? (
                  <TrendingUp color={trends.error_rate.direction === 'up' ? 'error' : 'success'} />
                ) : trends.error_rate.direction === 'down' ? (
                  <TrendingDown color={trends.error_rate.direction === 'down' ? 'success' : 'error'} />
                ) : (
                  <TrendingFlat color="action" />
                )}
                <Typography variant="body2" color="text.secondary" sx={{ ml: 1 }}>
                  {trends.error_rate.percentage.toFixed(1)}% {trends.error_rate.direction} from yesterday
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    );
  };

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const handleTimeRangeChange = (event) => {
    setTimeRange(event.target.value);
  };

  const formatDate = (dateStr) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString();
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" gutterBottom>
          Performance Monitoring
        </Typography>
        
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <FormControl variant="outlined" size="small" sx={{ minWidth: 120, mr: 2 }}>
            <InputLabel id="time-range-label">Time Range</InputLabel>
            <Select
              labelId="time-range-label"
              id="time-range"
              value={timeRange}
              onChange={handleTimeRangeChange}
              label="Time Range"
            >
              <MenuItem value="day">Today</MenuItem>
              <MenuItem value="week">Last 7 days</MenuItem>
              <MenuItem value="month">Last 30 days</MenuItem>
            </Select>
          </FormControl>
          
          <Tooltip title="Refresh data">
            <IconButton onClick={fetchData} color="primary">
              <Refresh />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>
      
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}
      
      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 5 }}>
          <CircularProgress />
        </Box>
      ) : metrics ? (
        <>
          {/* Summary Cards */}
          {calculateSummary()}
          
          {/* Tabs */}
          <Paper sx={{ mt: 3 }}>
            <Tabs
              value={tabValue}
              onChange={handleTabChange}
              indicatorColor="primary"
              textColor="primary"
              variant="fullWidth"
              sx={{ borderBottom: 1, borderColor: 'divider' }}
            >
              <Tab label="Request Volume" />
              <Tab label="Response Time" />
              <Tab label="Status Codes" />
              <Tab label="Endpoints" />
              <Tab label="Errors" />
              <Tab label="Model Info" />
            </Tabs>
            
            {/* Request Volume Tab */}
            {tabValue === 0 && (
              <Box sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Request Volume Over Time
                </Typography>
                
                <Box sx={{ height: 400, mt: 3 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={chartData}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis />
                      <RechartsTooltip />
                      <Legend />
                      <Bar dataKey="2xx" name="2xx Responses" fill={STATUS_COLORS['2xx']} stackId="a" />
                      <Bar dataKey="3xx" name="3xx Responses" fill={STATUS_COLORS['3xx']} stackId="a" />
                      <Bar dataKey="4xx" name="4xx Responses" fill={STATUS_COLORS['4xx']} stackId="a" />
                      <Bar dataKey="5xx" name="5xx Responses" fill={STATUS_COLORS['5xx']} stackId="a" />
                    </BarChart>
                  </ResponsiveContainer>
                </Box>
              </Box>
            )}
            
            {/* Response Time Tab */}
            {tabValue === 1 && (
              <Box sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Average Response Time
                </Typography>
                
                <Box sx={{ height: 400, mt: 3 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={chartData}
                      margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis label={{ value: 'Response Time (ms)', angle: -90, position: 'insideLeft' }} />
                      <RechartsTooltip formatter={(value) => `${(value * 1000).toFixed(2)} ms`} />
                      <Legend />
                      <Line type="monotone" dataKey="2xx_time" name="2xx Responses" stroke={STATUS_COLORS['2xx']} activeDot={{ r: 8 }} />
                      <Line type="monotone" dataKey="4xx_time" name="4xx Responses" stroke={STATUS_COLORS['4xx']} />
                      <Line type="monotone" dataKey="5xx_time" name="5xx Responses" stroke={STATUS_COLORS['5xx']} />
                    </LineChart>
                  </ResponsiveContainer>
                </Box>
              </Box>
            )}
            
            {/* Status Codes Tab */}
            {tabValue === 2 && (
              <Box sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Status Code Distribution
                </Typography>
                
                <Grid container spacing={3}>
                  <Grid item xs={12} md={6}>
                    <Box sx={{ height: 400, mt: 3 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie
                            data={statusDistribution}
                            cx="50%"
                            cy="50%"
                            labelLine={false}
                            outerRadius={150}
                            fill="#8884d8"
                            dataKey="value"
                            label={({ name, percent }) => `${name} (${(percent * 100).toFixed(1)}%)`}
                          >
                            {statusDistribution.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={STATUS_COLORS[entry.name] || COLORS[index % COLORS.length]} />
                            ))}
                          </Pie>
                          <RechartsTooltip formatter={(value) => value.toLocaleString()} />
                        </PieChart>
                      </ResponsiveContainer>
                    </Box>
                  </Grid>
                  
                  <Grid item xs={12} md={6}>
                    <TableContainer component={Paper} variant="outlined">
                      <Table>
                        <TableHead>
                          <TableRow>
                            <TableCell>Status Code</TableCell>
                            <TableCell align="right">Count</TableCell>
                            <TableCell align="right">Percentage</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {statusDistribution.map((row) => {
                            const total = statusDistribution.reduce((sum, item) => sum + item.value, 0);
                            const percentage = (row.value / total) * 100;
                            
                            return (
                              <TableRow key={row.name}>
                                <TableCell>
                                  <Chip 
                                    label={row.name} 
                                    size="small" 
                                    sx={{ 
                                      bgcolor: STATUS_COLORS[row.name] || COLORS[0],
                                      color: 'white'
                                    }} 
                                  />
                                </TableCell>
                                <TableCell align="right">{row.value.toLocaleString()}</TableCell>
                                <TableCell align="right">{percentage.toFixed(2)}%</TableCell>
                              </TableRow>
                            );
                          })}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </Grid>
                </Grid>
              </Box>
            )}
            
            {/* Endpoints Tab */}
            {tabValue === 3 && (
              <Box sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Top Endpoints
                </Typography>
                
                <TableContainer component={Paper} variant="outlined">
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Endpoint</TableCell>
                        <TableCell align="right">Requests</TableCell>
                        <TableCell align="right">Avg Response Time</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {metrics.summary.top_endpoints.map((endpoint) => (
                        <TableRow key={endpoint.endpoint}>
                          <TableCell>{endpoint.endpoint}</TableCell>
                          <TableCell align="right">{endpoint.count.toLocaleString()}</TableCell>
                          <TableCell align="right">{(endpoint.avg_response_time * 1000).toFixed(2)} ms</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Box>
            )}
            
            {/* Errors Tab */}
            {tabValue === 4 && (
              <Box sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Recent Errors
                </Typography>
                
                {metrics.summary.recent_errors.length > 0 ? (
                  <TableContainer component={Paper} variant="outlined">
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableCell>Timestamp</TableCell>
                          <TableCell>Endpoint</TableCell>
                          <TableCell>Status</TableCell>
                          <TableCell>Error</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {metrics.summary.recent_errors.map((error, index) => (
                          <TableRow key={index}>
                            <TableCell>{new Date(error.timestamp).toLocaleString()}</TableCell>
                            <TableCell>{error.endpoint}</TableCell>
                            <TableCell>
                              <Chip 
                                label={error.status_code} 
                                size="small" 
                                sx={{ 
                                  bgcolor: error.status_code >= 500 ? STATUS_COLORS['5xx'] : STATUS_COLORS['4xx'],
                                  color: 'white'
                                }} 
                              />
                            </TableCell>
                            <TableCell>{error.error}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                ) : (
                  <Alert severity="info">No errors recorded in the selected time period.</Alert>
                )}
              </Box>
            )}
            
            {/* Model Info Tab */}
            {tabValue === 5 && (
              <Box sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Model Information
                </Typography>
                
                {modelInfo ? (
                  <Grid container spacing={3}>
                    <Grid item xs={12} md={6}>
                      <Card variant="outlined">
                        <CardContent>
                          <Typography variant="h6" color="text.secondary" gutterBottom>
                            Model Configuration
                          </Typography>
                          
                          <Box sx={{ mt: 2 }}>
                            <Typography variant="body1">
                              <strong>Status:</strong> {modelInfo.loaded ? (
                                <Chip 
                                  icon={<CheckCircle />} 
                                  label="Loaded" 
                                  color="success" 
                                  size="small" 
                                  sx={{ ml: 1 }} 
                                />
                              ) : (
                                <Chip 
                                  icon={<ErrorIcon />} 
                                  label="Not Loaded" 
                                  color="error" 
                                  size="small" 
                                  sx={{ ml: 1 }} 
                                />
                              )}
                            </Typography>
                            
                            <Divider sx={{ my: 2 }} />
                            
                            <Typography variant="body1" gutterBottom>
                              <strong>Input Size:</strong> {modelInfo.input_size}
                            </Typography>
                            
                            <Typography variant="body1" gutterBottom>
                              <strong>Hidden Size:</strong> {modelInfo.hidden_size}
                            </Typography>
                            
                            <Typography variant="body1" gutterBottom>
                              <strong>Output Size:</strong> {modelInfo.output_size}
                            </Typography>
                            
                            <Divider sx={{ my: 2 }} />
                            
                            <Typography variant="body1" gutterBottom>
                              <strong>Optimized:</strong> {modelInfo.optimized ? 'Yes' : 'No'}
                            </Typography>
                            
                            <Typography variant="body1">
                              <strong>Cached:</strong> {modelInfo.cached ? 'Yes' : 'No'}
                            </Typography>
                          </Box>
                        </CardContent>
                      </Card>
                    </Grid>
                    
                    <Grid item xs={12} md={6}>
                      <Card variant="outlined">
                        <CardContent>
                          <Typography variant="h6" color="text.secondary" gutterBottom>
                            Performance Metrics
                          </Typography>
                          
                          <Box sx={{ mt: 2 }}>
                            <Typography variant="body1" gutterBottom>
                              <strong>Average Inference Time:</strong> 0.45 ms
                            </Typography>
                            
                            <Typography variant="body1" gutterBottom>
                              <strong>P95 Inference Time:</strong> 0.72 ms
                            </Typography>
                            
                            <Typography variant="body1" gutterBottom>
                              <strong>P99 Inference Time:</strong> 0.95 ms
                            </Typography>
                            
                            <Divider sx={{ my: 2 }} />
                            
                            <Typography variant="body1" gutterBottom>
                              <strong>Memory Usage:</strong> 128 MB
                            </Typography>
                            
                            <Typography variant="body1">
                              <strong>GPU Utilization:</strong> N/A
                            </Typography>
                          </Box>
                        </CardContent>
                      </Card>
                    </Grid>
                  </Grid>
                ) : (
                  <Alert severity="info">No model information available.</Alert>
                )}
              </Box>
            )}
          </Paper>
        </>
      ) : (
        <Alert severity="info">No performance data available.</Alert>
      )}
    </Container>
  );
};

export default PerformancePage; 