import React, { useState } from 'react';
import { 
  Container, Typography, Paper, Box, Button, 
  CircularProgress, Alert, List, ListItem, ListItemText,
  ListItemIcon, Grid, Card, CardContent
} from '@mui/material';
import { UploadFile, CheckCircle, Error } from '@mui/icons-material';
import { uploadData } from '../api/api';

const UploadPage = () => {
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [uploadResults, setUploadResults] = useState([]);

  const handleFileChange = (e) => {
    if (e.target.files) {
      setFiles(Array.from(e.target.files));
    }
  };

  const handleUpload = async () => {
    if (files.length === 0) return;
    
    setUploading(true);
    setError(null);
    setUploadResults([]);
    
    const results = [];
    
    for (const file of files) {
      try {
        const result = await uploadData(file);
        results.push({
          filename: file.name,
          success: true,
          message: result.message,
          processingTime: result.processing_time
        });
      } catch (err) {
        results.push({
          filename: file.name,
          success: false,
          message: err.message || 'Upload failed'
        });
        console.error(`Error uploading ${file.name}:`, err);
      }
    }
    
    setUploadResults(results);
    setUploading(false);
    
    // Check if any uploads failed
    const failures = results.filter(r => !r.success);
    if (failures.length > 0) {
      setError(`${failures.length} of ${results.length} files failed to upload.`);
    }
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom>
        Upload Data
      </Typography>
      
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}
      
      <Grid container spacing={3}>
        {/* Upload Form */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Upload Files
            </Typography>
            
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Upload CSV files with data for processing. Each file should contain
              data in the correct format for the model.
            </Typography>
            
            <Box sx={{ mt: 2 }}>
              <input
                accept=".csv,.txt,.json"
                style={{ display: 'none' }}
                id="file-upload"
                type="file"
                multiple
                onChange={handleFileChange}
              />
              <label htmlFor="file-upload">
                <Button
                  variant="outlined"
                  component="span"
                  startIcon={<UploadFile />}
                >
                  Select Files
                </Button>
              </label>
              
              {files.length > 0 && (
                <Typography variant="body2" sx={{ mt: 1 }}>
                  {files.length} file(s) selected
                </Typography>
              )}
              
              <Button
                variant="contained"
                onClick={handleUpload}
                disabled={uploading || files.length === 0}
                sx={{ mt: 2 }}
              >
                {uploading ? <CircularProgress size={24} /> : 'Upload'}
              </Button>
            </Box>
          </Paper>
        </Grid>
        
        {/* Upload Results */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Upload Results
              </Typography>
              
              {uploadResults.length > 0 ? (
                <List>
                  {uploadResults.map((result, index) => (
                    <ListItem key={index}>
                      <ListItemIcon>
                        {result.success ? (
                          <CheckCircle color="success" />
                        ) : (
                          <Error color="error" />
                        )}
                      </ListItemIcon>
                      <ListItemText
                        primary={result.filename}
                        secondary={
                          result.success
                            ? `${result.message} (${result.processingTime.toFixed(4)}s)`
                            : result.message
                        }
                      />
                    </ListItem>
                  ))}
                </List>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  No files uploaded yet.
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Container>
  );
};

export default UploadPage; 