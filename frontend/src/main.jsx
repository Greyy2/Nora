import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import 'bootstrap/dist/css/bootstrap.min.css'
import 'bootstrap-icons/font/bootstrap-icons.css'
import './css/styles.css'
import { ThemeProvider } from './user/ThemeContext';

const syncViewportCssVars = () => {
    if (typeof window === 'undefined') return;
    const vv = window.visualViewport;
    const viewportHeight = Math.max(320, Math.round(vv?.height || window.innerHeight || 0));
    const viewportWidth = Math.max(320, Math.round(vv?.width || window.innerWidth || 0));
    document.documentElement.style.setProperty('--vh', `${viewportHeight}px`);
    document.documentElement.style.setProperty('--vw', `${viewportWidth}px`);
};

if (typeof window !== 'undefined') {
    syncViewportCssVars();
    window.addEventListener('resize', syncViewportCssVars, { passive: true });
    window.addEventListener('orientationchange', syncViewportCssVars, { passive: true });
    window.visualViewport?.addEventListener('resize', syncViewportCssVars, { passive: true });
    window.visualViewport?.addEventListener('scroll', syncViewportCssVars, { passive: true });
}


ReactDOM.createRoot(document.getElementById('root')).render(
    <React.StrictMode>
        <ThemeProvider>
            <App />
        </ThemeProvider>
    </React.StrictMode>,
)
