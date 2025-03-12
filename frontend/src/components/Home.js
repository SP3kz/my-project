import React, { useState, useEffect } from 'react';
import { Container, Typography, Paper, Box, Grid, Card, CardContent, Button } from '@mui/material';
import { Link as RouterLink } from 'react-router-dom';
import { healthCheck, getModelInfo } from '../api/api';

const Home = () => {
  const [health, setHealth] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const healthData = await healthCheck();
        const modelData = await getModelInfo();
        
        setHealth(healthData);
        setModelInfo(modelData);
        setError(null);
      } catch (err) {
        setError('Failed to fetch data from the API. Make sure the backend is running.');
        console.error('Error fetching data:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom>
        ML Project Dashboard
      </Typography>
      
      {error && (
        <Paper sx={{ p: 2, mb: 3, bgcolor: '#ffebee' }}>
          <Typography color="error">{error}</Typography>
        </Paper>
      )}
      
      <Grid container spacing={3}>
        {/* API Status */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                API Status
              </Typography>
              {loading ? (
                <Typography>Loading...</Typography>
              ) : health ? (
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Box
                    sx={{
                      width: 12,
                      height: 12,
                      borderRadius: '50%',
                      bgcolor: health.status === 'healthy' ? 'success.main' : 'error.main',
                    }}
                  />
                  <Typography>
                    {health.status === 'healthy' ? 'API is running' : 'API has issues'}
                  </Typography>
                </Box>
              ) : (
                <Typography color="error">Unable to connect to API</Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
        
        {/* Model Status */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Model Status
              </Typography>
              {loading ? (
                <Typography>Loading...</Typography>
              ) : modelInfo ? (
                <Box>
                  <Typography>
                    Model Loaded: {modelInfo.loaded ? 'Yes' : 'No'}
                  </Typography>
                  {modelInfo.loaded && (
                    <>
                      <Typography>
                        Input Size: {modelInfo.input_size}
                      </Typography>
                      <Typography>
                        Hidden Size: {modelInfo.hidden_size}
                      </Typography>
                      <Typography>
                        Output Size: {modelInfo.output_size}
                      </Typography>
                      <Typography>
                        Optimized: {modelInfo.optimized ? 'Yes' : 'No'}
                      </Typography>
                      <Typography>
                        Cached: {modelInfo.cached ? 'Yes' : 'No'}
                      </Typography>
                    </>
                  )}
                </Box>
              ) : (
                <Typography color="error">Unable to get model info</Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
        
        {/* Quick Actions */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Quick Actions
            </Typography>
            <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
              <Button
                variant="contained"
                component={RouterLink}
                to="/model"
              >
                Load Model
              </Button>
              <Button
                variant="contained"
                component={RouterLink}
                to="/predict"
                disabled={!modelInfo?.loaded}
              >
                Make Predictions
              </Button>
              <Button
                variant="contained"
                component={RouterLink}
                to="/upload"
              >
                Upload Data
              </Button>
              <Button
                variant="contained"
                component={RouterLink}
                to="/performance"
              >
                View Performance
              </Button>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default Home; 