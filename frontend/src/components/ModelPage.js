import React, { useState, useEffect } from 'react';
import { 
  Container, Typography, Paper, Box, Button, TextField, 
  FormControlLabel, Checkbox, CircularProgress, Alert, Grid, Card, CardContent 
} from '@mui/material';
import { getModelInfo, loadModel } from '../api/api';

const ModelPage = () => {
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [loadingModel, setLoadingModel] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  
  // Form state
  const [inputSize, setInputSize] = useState(20);
  const [hiddenSize, setHiddenSize] = useState(64);
  const [outputSize, setOutputSize] = useState(2);
  const [optimize, setOptimize] = useState(true);
  const [cache, setCache] = useState(true);

  useEffect(() => {
    fetchModelInfo();
  }, []);

  const fetchModelInfo = async () => {
    try {
      setLoading(true);
      const data = await getModelInfo();
      setModelInfo(data);
      
      // Update form with current model info if loaded
      if (data.loaded) {
        setInputSize(data.input_size);
        setHiddenSize(data.hidden_size);
        setOutputSize(data.output_size);
      }
      
      setError(null);
    } catch (err) {
      setError('Failed to fetch model info. Make sure the backend is running.');
      console.error('Error fetching model info:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleLoadModel = async (e) => {
    e.preventDefault();
    
    try {
      setLoadingModel(true);
      setSuccess(null);
      setError(null);
      
      const config = {
        input_size: parseInt(inputSize),
        hidden_size: parseInt(hiddenSize),
        output_size: parseInt(outputSize),
        optimize,
        cache
      };
      
      const result = await loadModel(config);
      
      setSuccess(`Model loaded successfully. ${result.initialized_new ? 'New model initialized.' : 'Existing model loaded.'}`);
      
      // Refresh model info
      await fetchModelInfo();
    } catch (err) {
      setError('Failed to load model. Check the console for details.');
      console.error('Error loading model:', err);
    } finally {
      setLoadingModel(false);
    }
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom>
        Model Management
      </Typography>
      
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}
      
      {success && (
        <Alert severity="success" sx={{ mb: 3 }}>
          {success}
        </Alert>
      )}
      
      <Grid container spacing={3}>
        {/* Current Model Info */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Current Model Status
              </Typography>
              
              {loading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center' }}>
                  <CircularProgress />
                </Box>
              ) : modelInfo ? (
                <Box>
                  <Typography variant="body1" gutterBottom>
                    Model Loaded: {modelInfo.loaded ? 'Yes' : 'No'}
                  </Typography>
                  
                  {modelInfo.loaded && (
                    <>
                      <Typography variant="body1" gutterBottom>
                        Input Size: {modelInfo.input_size}
                      </Typography>
                      <Typography variant="body1" gutterBottom>
                        Hidden Size: {modelInfo.hidden_size}
                      </Typography>
                      <Typography variant="body1" gutterBottom>
                        Output Size: {modelInfo.output_size}
                      </Typography>
                      <Typography variant="body1" gutterBottom>
                        Optimized: {modelInfo.optimized ? 'Yes' : 'No'}
                      </Typography>
                      <Typography variant="body1" gutterBottom>
                        Cached: {modelInfo.cached ? 'Yes' : 'No'}
                      </Typography>
                    </>
                  )}
                  
                  <Button 
                    variant="outlined" 
                    onClick={fetchModelInfo} 
                    sx={{ mt: 2 }}
                  >
                    Refresh
                  </Button>
                </Box>
              ) : (
                <Typography color="error">
                  Unable to get model info
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
        
        {/* Load Model Form */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Load Model
            </Typography>
            
            <Box component="form" onSubmit={handleLoadModel} noValidate>
              <TextField
                margin="normal"
                required
                fullWidth
                id="inputSize"
                label="Input Size"
                name="inputSize"
                type="number"
                value={inputSize}
                onChange={(e) => setInputSize(e.target.value)}
              />
              
              <TextField
                margin="normal"
                required
                fullWidth
                id="hiddenSize"
                label="Hidden Size"
                name="hiddenSize"
                type="number"
                value={hiddenSize}
                onChange={(e) => setHiddenSize(e.target.value)}
              />
              
              <TextField
                margin="normal"
                required
                fullWidth
                id="outputSize"
                label="Output Size"
                name="outputSize"
                type="number"
                value={outputSize}
                onChange={(e) => setOutputSize(e.target.value)}
              />
              
              <FormControlLabel
                control={
                  <Checkbox
                    checked={optimize}
                    onChange={(e) => setOptimize(e.target.checked)}
                    name="optimize"
                    color="primary"
                  />
                }
                label="Optimize for inference"
              />
              
              <FormControlLabel
                control={
                  <Checkbox
                    checked={cache}
                    onChange={(e) => setCache(e.target.checked)}
                    name="cache"
                    color="primary"
                  />
                }
                label="Cache model weights"
              />
              
              <Button
                type="submit"
                fullWidth
                variant="contained"
                sx={{ mt: 3, mb: 2 }}
                disabled={loadingModel}
              >
                {loadingModel ? <CircularProgress size={24} /> : 'Load Model'}
              </Button>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default ModelPage; 