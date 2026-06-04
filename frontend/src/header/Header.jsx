import React, { useState, useRef, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../user/AuthContext';
import { useTheme } from '../user/ThemeContext';
import UserDropdown from './components/UserDropdown';
import BrandIcon from '../components/BrandIcon';
import './Header.css';

const Header = () => {
    const { user, logout } = useAuth();
    const { theme, toggleTheme } = useTheme();
    const [showDropdown, setShowDropdown] = useState(false);
    const dropdownRef = useRef(null);
    const navigate = useNavigate();
    const location = useLocation();

    useEffect(() => {
        const handleClickOutside = (event) => {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
                setShowDropdown(false);
            }
        };

        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    const handleLogout = () => {
        logout();
        navigate('/home');
    };

    // Ẩn header ở các trang đã có navbar riêng
    const hideHeaderPaths = ['/login', '/home', '/backtest', '/trading', '/about', '/client'];
    if (hideHeaderPaths.includes(location.pathname)) return null;

    const getInitials = (name) => {
        if (!name) return 'U';
        return name.split(' ')
            .map(n => n[0])
            .join('')
            .toUpperCase()
            .slice(0, 2);
    };

    return (
        <header className="app-header">
            <div className="header-content">
                <div className="header-left">
                    <h1 className="header-logo" onClick={() => navigate('/home')} style={{ cursor: 'pointer' }}>
                        <BrandIcon size={22} className="header-logo-icon" />
                        Nora <span className="logo-highlight">Trading</span>
                    </h1>
                </div>

                <nav className="header-nav">
                    <div className="nav-links">
                        <button
                            className="nav-button"
                            onClick={() => navigate('/home')}
                            title="Trang chủ"
                        >
                            <i className="bi bi-house-door"></i>
                            <span>Home</span>
                        </button>
                        <button
                            className="nav-button"
                            onClick={() => navigate('/backtest')}
                            title="Backtest"
                        >
                            <i className="bi bi-graph-up"></i>
                            <span>Backtest</span>
                        </button>
                        <button
                            className="nav-button"
                            onClick={() => navigate('/trading')}
                            title="Trading"
                        >
                            <i className="bi bi-currency-exchange"></i>
                            <span>Trading</span>
                        </button>
                        <button
                            className="nav-button"
                            onClick={() => navigate('/client')}
                            title="Client"
                        >
                            <i className="bi bi-cpu-fill"></i>
                            <span>Client</span>
                        </button>
                    </div>

                    <div className="header-actions-right">
                        {/* Theme Toggle */}
                        <button
                            className="nav-theme-toggle"
                            onClick={toggleTheme}
                            aria-label="Toggle Theme"
                            title={theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
                        >
                            {theme === 'dark' ? (
                                <i className="bi bi-moon-stars-fill"></i>
                            ) : (
                                <i className="bi bi-sun-fill"></i>
                            )}
                        </button>

                        {/* Login or User Profile */}
                        {user ? (
                            <UserDropdown />
                        ) : (
                            <button
                                className="nav-button nav-login-button"
                                onClick={() => navigate('/login')}
                                title="Đăng nhập"
                            >
                                <i className="bi bi-box-arrow-in-right"></i>
                                <span>Đăng nhập</span>
                            </button>
                        )}
                    </div>
                </nav>

                {/* Removed old header-right section */}
            </div>
        </header>
    );
};

export default Header;
