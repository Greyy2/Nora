import React, { useState, useEffect } from 'react';
import './Toast.css';

const Toast = () => {
    const [toast, setToast] = useState(null);

    useEffect(() => {
        const checkToast = () => {
            const message = sessionStorage.getItem('loginSuccess');
            if (message) {
                setToast({ message, type: 'success' });
                sessionStorage.removeItem('loginSuccess');

                setTimeout(() => {
                    setToast(null);
                }, 3000);
            }
        };

        // Check immediately
        checkToast();

        // Also listen for storage events (optional but good for multi-tab)
        window.addEventListener('storage', checkToast);

        // Custom event for same-tab updates if needed
        window.addEventListener('showToast', (e) => {
            setToast({ message: e.detail.message, type: e.detail.type || 'success' });
            setTimeout(() => setToast(null), 3000);
        });

        return () => {
            window.removeEventListener('storage', checkToast);
            window.removeEventListener('showToast', checkToast);
        };
    }, []);

    if (!toast) return null;

    return (
        <div className={`toast-container ${toast ? 'active' : ''}`}>
            <div className={`toast-content ${toast.type}`}>
                <i className={`bi ${toast.type === 'success' ? 'bi-check-circle-fill' : 'bi-exclamation-triangle-fill'}`}></i>
                <span>{toast.message}</span>
                <div className="toast-progress"></div>
            </div>
        </div>
    );
};

export default Toast;
