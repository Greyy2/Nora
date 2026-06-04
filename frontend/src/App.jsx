import React from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate, useLocation } from 'react-router-dom'
import { AuthProvider } from './user/AuthContext'
import ProtectedRoute from './header/ProtectedRoute'
import Header from './header/Header'
import Toast from './header/Toast'
import ParticleBackground from './components/ParticleBackground'

import Login from './pages/Login';
import Home from './pages/Home';
import About from './pages/About';
import SingleCore from './pages/SingleCore';
import Backtest from './pages/Backtest';
import Client from './pages/Client';
import Settings from './pages/Settings';
import Help from './pages/Help';
import Profile from './pages/Profile';

function App() {
    return (
        <AuthProvider>
            <Router future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
                <AppContent />
            </Router>
        </AuthProvider>
    )
}

function AppContent() {
    const location = useLocation();
    const isClientRoute = location.pathname.startsWith('/client') || location.pathname.startsWith('/quanta-app');

    return (
        <div className={isClientRoute ? "client-root" : "grey-root"}>
            {!isClientRoute && <ParticleBackground />}
            {!isClientRoute && <Header />}
            <Toast />
            <Routes>
                <Route path="/" element={<Navigate to="/home" replace />} />
                <Route path="/login" element={<Login />} />
                <Route path="/home" element={<Home />} />
                <Route path="/about" element={<About />} />
                <Route
                    path="/backtest"
                    element={
                        <ProtectedRoute>
                            <Backtest />
                        </ProtectedRoute>
                    }
                />
                <Route
                    path="/client"
                    element={
                        <ProtectedRoute>
                            <Client />
                        </ProtectedRoute>
                    }
                />
                <Route
                    path="/trading"
                    element={
                        <ProtectedRoute>
                            <SingleCore />
                        </ProtectedRoute>
                    }
                />
                <Route
                    path="/settings"
                    element={
                        <ProtectedRoute>
                            <Settings />
                        </ProtectedRoute>
                    }
                />
                <Route
                    path="/help"
                    element={
                        <ProtectedRoute>
                            <Help />
                        </ProtectedRoute>
                    }
                />
                <Route path="/profile" element={<ProtectedRoute><Profile /></ProtectedRoute>} />
                {/* Redirect legacy quanta-app paths to the new /client path */}
                <Route path="/quanta-app/*" element={<Navigate to="/client" replace />} />
            </Routes>
        </div>
    )
}

export default App
