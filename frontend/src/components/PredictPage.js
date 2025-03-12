import React, { useState, useEffect } from 'react';
import { 
  Container, Typography, Paper, Box, Button, TextField, 
  CircularProgress, Alert, Grid, Card, CardContent, Table,
  TableBody, TableCell, TableContainer, TableHead, TableRow
} from '@mui/material';
import { getModelInfo, predict } from '../api/api';

const PredictPage = () => {
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState(null);
  const [inputData, setInputData] = useState('');
  const [predictions, setPredictions] = useState(null);
  const [processingTime, setProcessingTime] = useState(null);

  useEffect(() => {
    fetchModelInfo();
  }, []);

  const fetchModelInfo = async () => {
    try {
      setLoading(true);
      const data = await getModelInfo();
      setModelInfo(data);
      setError(null);
    } catch (err) {
      setError('Failed to fetch model info. Make sure the backend is running.');
      console.error('Error fetching model info:', err);
    } finally {
      setLoading(false);
    }
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    
    try {
      setPredicting(true);
      setError(null);
      setPredictions(null);
      setProcessingTime(null);
      
      // Parse input data
      let features = [];
      try {
        // Try to parse as JSON
        features = JSON.parse(inputData);
        
        // Ensure it's an array of arrays
        if (!Array.isArray(features)) {
          features = [features];
        }
        
        // If it's an array of numbers, wrap it in another array
        if (features.length > 0 && !Array.isArray(features[0])) {
          features = [features];
        }
      } catch (parseError) {
        // If JSON parsing fails, try to parse as comma-separated values
        const rows = inputData.trim().split('\n');
        features = rows.map(row => 
          row.split(',').map(val => parseFloat(val.trim()))
        );
      }
      
      // Validate input dimensions
      if (modelInfo.loaded && features[0].length !== modelInfo.input_size) {
        throw new Error(`Input size mismatch. Expected ${modelInfo.input_size} features, but got ${features[0].length}.`);
      }
      
      const result = await predict(features);
      setPredictions(result.predictions);
      setProcessingTime(result.processing_time);
    } catch (err) {
      setError(`Prediction failed: ${err.message}`);
      console.error('Error making prediction:', err);
    } finally {
      setPredicting(false);
    }
  };

  const generateSampleData = () => {
    if (!modelInfo || !modelInfo.loaded) return;
    
    // Generate random data based on the model's input size
    const sampleData = [];
    for (let i = 0; i < 5; i++) {
      const row = [];
      for (let j = 0; j < modelInfo.input_size; j++) {
        row.push(parseFloat((Math.random() * 2 - 1).toFixed(4)));
      }
      sampleData.push(row);
    }
    
    setInputData(JSON.stringify(sampleData, null, 2));
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom>
        Make Predictions
      </Typography>
      
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}
      
      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center' }}>
          <CircularProgress />
        </Box>
      ) : !modelInfo?.loaded ? (
        <Alert severity="warning" sx={{ mb: 3 }}>
          No model is loaded. Please load a model first.
        </Alert>
      ) : (
        <Grid container spacing={3}>
          {/* Input Form */}
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Input Data
              </Typography>
              
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Enter your input data as JSON array or comma-separated values.
                Each row should have {modelInfo.input_size} features.
              </Typography>
              
              <Box component="form" onSubmit={handlePredict} noValidate>
                <TextField
                  margin="normal"
                  required
                  fullWidth
                  id="inputData"
                  label="Input Data"
                  name="inputData"
                  multiline
                  rows={10}
                  value={inputData}
                  onChange={(e) => setInputData(e.target.value)}
                  placeholder={`Example: [[0.1, 0.2, ...], [0.3, 0.4, ...]]`}
                />
                
                <Box sx={{ display: 'flex', gap: 2, mt: 2 }}>
                  <Button
                    variant="outlined"
                    onClick={generateSampleData}
                  >
                    Generate Sample Data
                  </Button>
                  
                  <Button
                    type="submit"
                    variant="contained"
                    disabled={predicting || !inputData.trim()}
                  >
                    {predicting ? <CircularProgress size={24} /> : 'Predict'}
                  </Button>
                </Box>
              </Box>
            </Paper>
          </Grid>
          
          {/* Results */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Prediction Results
                </Typography>
                
                {predicting ? (
                  <Box sx={{ display: 'flex', justifyContent: 'center' }}>
                    <CircularProgress />
                  </Box>
                ) : predictions ? (
                  <>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Processing time: {processingTime.toFixed(6)} seconds
                    </Typography>
                    
                    <TableContainer component={Paper} sx={{ mt: 2 }}>
                      <Table size="small">
                        <TableHead>
                          <TableRow>
                            <TableCell>Sample</TableCell>
                            {Array.from({ length: modelInfo.output_size }).map((_, i) => (
                              <TableCell key={i}>Output {i+1}</TableCell>
                            ))}
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {predictions.map((prediction, i) => (
                            <TableRow key={i}>
                              <TableCell>{i+1}</TableCell>
                              {prediction.map((value, j) => (
                                <TableCell key={j}>{value.toFixed(6)}</TableCell>
                              ))}
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </>
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    No predictions yet. Enter input data and click "Predict".
                  </Typography>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}
    </Container>
  );
};

export default PredictPage; 