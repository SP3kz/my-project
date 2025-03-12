import React from 'react';
import { AppBar, Toolbar, Typography, Button, Box } from '@mui/material';
import { Link as RouterLink } from 'react-router-dom';

const Header = () => {
  return (
    <AppBar position="static">
      <Toolbar>
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          ML Project
        </Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button color="inherit" component={RouterLink} to="/">
            Home
          </Button>
          <Button color="inherit" component={RouterLink} to="/model">
            Model
          </Button>
          <Button color="inherit" component={RouterLink} to="/predict">
            Predict
          </Button>
          <Button color="inherit" component={RouterLink} to="/upload">
            Upload
          </Button>
          <Button color="inherit" component={RouterLink} to="/performance">
            Performance
          </Button>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Header; 