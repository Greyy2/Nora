import React, { useEffect } from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from '../user/AuthContext';

const ProtectedRoute = ({ children }) => {
    const { isAuthenticated, loading } = useAuth();
    const location = useLocation();

    useEffect(() => {
        // Lưu URL hiện tại để redirect về sau khi login
        if (!isAuthenticated && !loading) {
            localStorage.setItem('redirectAfterLogin', location.pathname + location.search);
        }
    }, [isAuthenticated, loading, location]);

    if (loading) {
        return (
            <div style={{
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                height: '100vh',
                background: 'var(--tv-bg-primary)'
            }}>
                <div className="spinner" style={{
                    border: '3px solid rgba(255, 255, 255, 0.1)',
                    borderTop: '3px solid var(--tv-accent-blue)',
                    borderRadius: '50%',
                    width: '50px',
                    height: '50px',
                    animation: 'spin 1s linear infinite'
                }}></div>
            </div>
        );
    }

    if (!isAuthenticated) {
        return <Navigate to="/login" replace />;
    }

    return children;
};

export default ProtectedRoute;
