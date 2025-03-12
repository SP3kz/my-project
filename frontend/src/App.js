import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { Box } from '@mui/material';

// Layout components
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';

// Page components
import Dashboard from './components/Dashboard';
import ModelTraining from './components/ModelTraining';
import DatasetManagement from './components/DatasetManagement';
import Predictions from './components/Predictions';
import PerformancePage from './components/PerformancePage';
import Settings from './components/Settings';
import NotFound from './components/NotFound';

// Auth components
import Login from './components/auth/Login';
import Register from './components/auth/Register';
import ProtectedRoute from './components/auth/ProtectedRoute';

// Create theme
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#3f51b5',
    },
    secondary: {
      main: '#f50057',
    },
  },
});

function App() {
  const [open, setOpen] = React.useState(true);
  const toggleDrawer = () => {
    setOpen(!open);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Routes>
          {/* Public routes */}
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          
          {/* Protected routes with layout */}
          <Route path="/" element={
            <ProtectedRoute>
              <Box sx={{ display: 'flex' }}>
                <Navbar open={open} toggleDrawer={toggleDrawer} />
                <Sidebar open={open} />
                <Box
                  component="main"
                  sx={{
                    backgroundColor: (theme) =>
                      theme.palette.mode === 'light'
                        ? theme.palette.grey[100]
                        : theme.palette.grey[900],
                    flexGrow: 1,
                    height: '100vh',
                    overflow: 'auto',
                    pt: 8, // Offset for navbar
                  }}
                >
                  <Routes>
                    <Route path="/" element={<Dashboard />} />
                    <Route path="/training" element={<ModelTraining />} />
                    <Route path="/datasets" element={<DatasetManagement />} />
                    <Route path="/predictions" element={<Predictions />} />
                    <Route path="/performance" element={<PerformancePage />} />
                    <Route path="/settings" element={<Settings />} />
                    <Route path="*" element={<NotFound />} />
                  </Routes>
                </Box>
              </Box>
            </ProtectedRoute>
          } />
          
          {/* Catch all route */}
          <Route path="*" element={<NotFound />} />
        </Routes>
      </Router>
    </ThemeProvider>
  );
}

export default App;
