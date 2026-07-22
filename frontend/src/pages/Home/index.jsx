import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useHomeLogic } from './logic';
import { useAuth } from '../../user/AuthContext';
import { useTheme } from '../../user/ThemeContext';
import UserDropdown from '../../header/components/UserDropdown';
import BrandIcon from '../../components/BrandIcon';
import './style.css';

const preloadBacktest = () => import('../Backtest');
const preloadClient = () => import('../Client');

const Home = () => {
    const {
        user,
        activeSection,
        scrollToSection,
        toggleMenu,
        showSystemModal,
        openSystemModal,
        closeSystemModal
    } = useHomeLogic();

    const { theme, toggleTheme } = useTheme();
    const [showStartMenu, setShowStartMenu] = useState(false);
    const navigate = useNavigate();

    const handleStartClick = () => {
        if (user) {
            setShowStartMenu(!showStartMenu);
            preloadBacktest();
            preloadClient();
        } else {
            navigate('/login');
        }
    };

    const handleMenuItemClick = (path) => {
        setShowStartMenu(false);
        navigate(path);
    };

    return (
        <div className={`home-container home-page ${theme}`}>
            {/* Header Navigation */}
            <header id="header" className="home-header">
                <div className="container">
                    <div className="logo">
                        <BrandIcon size={22} className="logo-icon" />
                        <span className="logo-text">noraquantengine</span>
                    </div>

                    <nav>
                        <ul className="nav-menu">
                            <li>
                                <button
                                    type="button"
                                    onClick={() => scrollToSection('home')}
                                    className={`nav-link nav-link-button ${activeSection === 'home' ? 'active' : ''}`}
                                >
                                    Trang chủ
                                </button>
                            </li>
                            <li>
                                <Link
                                    to="/about"
                                    className="nav-link"
                                >
                                    Về chúng tôi
                                </Link>
                            </li>
                        </ul>
                    </nav>

                    <div className="header-actions">
                        {/* User Menu / Login */}
                        {user ? (
                            <>
                                {/* User Dropdown */}
                                <UserDropdown />
                            </>
                        ) : (
                            <Link to="/login" className="btn btn-login-grad">
                                <i className="bi bi-box-arrow-in-right"></i>
                                <span>Đăng nhập</span>
                            </Link>
                        )}

                        {/* Theme Toggle */}
                        <button className="marketing-theme-toggle-btn" onClick={toggleTheme} aria-label="Toggle Theme">
                            {theme === 'dark' ? (
                                <i className="bi bi-moon-stars-fill"></i>
                            ) : (
                                <i className="bi bi-sun-fill"></i>
                            )}
                        </button>
                    </div>
                </div>
            </header>

            {/* Slide 1: Welcome */}
            <section id="home" className="slide welcome-slide">
                <div className="hero-background">
                    <div className="gradient-orb orb-1"></div>
                    <div className="gradient-orb orb-2"></div>
                    <div className="gradient-orb orb-3"></div>
                    <div className="gradient-orb orb-4"></div>
                    <div className="gradient-orb orb-5"></div>
                </div>
                <div className="slide-content">
                    <div className="welcome-text">
                        <h1 className="main-title">noraquantengine</h1>
                        <h2 className="subtitle">Professional Trading Platform</h2>
                        <p className="description">Advanced Strategy Backtester & Optimization Engine - Production-grade platform với TradingView matching, high-speed multiprocessing optimization, và comprehensive risk metrics</p>
                        
                        <div className="cta-buttons">
                            <div className="start-menu-wrapper">
                                <button className="btn btn-primary btn-hero" onClick={handleStartClick}>
                                    <span className="btn-icon">🚀</span>
                                    <span>Bắt đầu ngay</span>
                                    {user && <i className="bi bi-chevron-down start-chevron"></i>}
                                </button>
                                
                                {showStartMenu && user && (
                                    <>
                                        <div className="start-menu-backdrop" onClick={() => setShowStartMenu(false)}></div>
                                        <div className="start-dropdown-menu">
                                            <div className="start-menu-item" onClick={() => handleMenuItemClick('/trading')}>
                                                <div className="start-menu-icon trading">
                                                    <i className="bi bi-graph-up-arrow"></i>
                                                </div>
                                                <div className="start-menu-content">
                                                    <span className="start-menu-title">Trading</span>
                                                    <span className="start-menu-desc">Giao dịch real-time</span>
                                                </div>
                                                <i className="bi bi-chevron-right"></i>
                                            </div>
                                            <div className="start-menu-item" onMouseEnter={preloadBacktest} onClick={() => handleMenuItemClick('/backtest')}>
                                                <div className="start-menu-icon backtest">
                                                    <i className="bi bi-lightning-charge-fill"></i>
                                                </div>
                                                <div className="start-menu-content">
                                                    <span className="start-menu-title">Backtest</span>
                                                    <span className="start-menu-desc">Kiểm tra chiến lược</span>
                                                </div>
                                                <i className="bi bi-chevron-right"></i>
                                            </div>
                                            <div className="start-menu-item" onMouseEnter={preloadClient} onClick={() => handleMenuItemClick('/client')}>
                                                <div className="start-menu-icon client">
                                                    <i className="bi bi-cpu-fill"></i>
                                                </div>
                                                <div className="start-menu-content">
                                                    <span className="start-menu-title">Client</span>
                                                    <span className="start-menu-desc">AI Factor Discovery</span>
                                                </div>
                                                <i className="bi bi-chevron-right"></i>
                                            </div>
                                        </div>
                                    </>
                                )}
                            </div>
                        </div>
                    </div>
                    <div className="welcome-stats">
                        <div className="stat-card">
                            <div className="stat-icon">⚡</div>
                            <div className="stat-number">23K</div>
                            <div className="stat-label">Configs/giây</div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-icon">🔥</div>
                            <div className="stat-number">120</div>
                            <div className="stat-label">Backtests/giây</div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-icon">🎯</div>
                            <div className="stat-number">39</div>
                            <div className="stat-label">CPU Workers</div>
                        </div>
                    </div>
                </div>
            </section>

            {/* Click outside to close menu */}
            {showStartMenu && <div className="start-menu-backdrop" onClick={() => setShowStartMenu(false)}></div>}

            {/* System Selection Modal */}
            {showSystemModal && (
                <div className="system-modal-overlay" onClick={closeSystemModal}>
                    <div className="system-modal-content" onClick={(e) => e.stopPropagation()}>
                        <button className="system-modal-close" onClick={closeSystemModal}>✕</button>
                        <h2>Chọn Hệ Thống</h2>
                        <p className="system-modal-desc">Chọn hệ thống bạn muốn sử dụng</p>
                        <div className="system-modal-cards">
                            <Link to="/trading" className="system-modal-card" onClick={closeSystemModal}>
                                <div className="system-card-icon">📈</div>
                                <h3>Trading</h3>
                                <p>Hệ thống giao dịch thực tế với quản lý portfolio và tín hiệu real-time</p>
                                <span className="system-card-btn">Vào Trading →</span>
                            </Link>
                            <Link to="/backtest" className="system-modal-card premium" onClick={closeSystemModal}>
                                <div className="system-card-icon">🔬</div>
                                <h3>Backtest</h3>
                                <p>Kiểm tra chiến lược với dữ liệu lịch sử, tối ưu hóa tham số tự động</p>
                                <span className="system-card-btn premium">Vào Backtest →</span>
                            </Link>
                            <Link to="/client" className="system-modal-card" onClick={closeSystemModal}>
                                <div className="system-card-icon">🤖</div>
                                <h3>Client</h3>
                                <p>AI Factor Discovery & Client System Integration</p>
                                <span className="system-card-btn">Vào Client →</span>
                            </Link>
                        </div>
                    </div>
                </div>
            )}
        </div >
    );
};

export default Home;
