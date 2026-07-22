import React from 'react';
import { Link } from 'react-router-dom';
import { useLoginLogic } from './logic';
import { useTheme } from '../../user/ThemeContext';
import './style.css';

const Login = () => {
    const {
        formData,
        status,
        showPassword,
        handleChange,
        togglePasswordVisibility,
        handleLogin
    } = useLoginLogic();

    const { theme, toggleTheme } = useTheme();

    return (
        <div className={`login-page-wrapper ${theme}`}>
            {/* Back to Home Button */}
            <Link to="/home" className="back-to-home-btn" aria-label="Back to Home">
                <i className="bi bi-arrow-left"></i>
                <span>Back</span>
            </Link>

            {/* Theme Toggle Button */}
            <button className="login-theme-toggle-btn" onClick={toggleTheme} aria-label="Toggle Theme">
                {theme === 'dark' ? (
                    <i className="bi bi-moon-stars-fill"></i>
                ) : (
                    <i className="bi bi-sun-fill"></i>
                )}
            </button>

            {/* Animated Wave Orbs Background */}
            <div className="hero-background">
                <div className="gradient-orb orb-1"></div>
                <div className="gradient-orb orb-2"></div>
                <div className="gradient-orb orb-3"></div>
                <div className="gradient-orb orb-4"></div>
                <div className="gradient-orb orb-5"></div>
            </div>

            <div className="login-split-container">
                {/* LEFT SIDE: Branding & info */}
                <div className="login-left-panel">
                    <div className="branding-content">
                        <h1 className="brand-title">
                            Nora <span className="brand-highlight">Trading</span>
                        </h1>

                        <div className="feature-list">
                            <div className="feature-item">
                                <div className="feature-bullet">
                                    <span className="bullet-circle"></span>
                                </div>
                                <div className="feature-text">
                                    <h3>Backtest Chiến lược</h3>
                                    <p>Chạy backtest nhanh chóng trên dữ liệu lịch sử chất lượng cao</p>
                                </div>
                            </div>

                            <div className="feature-item">
                                <div className="feature-bullet">
                                    <span className="bullet-circle"></span>
                                </div>
                                <div className="feature-text">
                                    <h3>Optimize Thông minh</h3>
                                    <p>Tối ưu hóa tham số tự động với thuật toán tiên tiến</p>
                                </div>
                            </div>

                            <div className="feature-item">
                                <div className="feature-bullet">
                                    <span className="bullet-circle"></span>
                                </div>
                                <div className="feature-text">
                                    <h3>Độ chính xác Cao</h3>
                                    <p>Phân tích hiệu suất chi tiết với metrics chuyên sâu</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* RIGHT SIDE: Login Form */}
                <div className="login-right-panel">
                    <div className="login-form-container">
                        <div className="form-header">
                            <h2>Welcome Trader</h2>
                            <p>Đăng nhập để truy cập hệ thống</p>
                        </div>

                        <form className="modern-form" onSubmit={handleLogin} autoComplete="off">
                            <div className="input-group-modern">
                                <label htmlFor="username">Tên đăng nhập hoặc Email</label>
                                <div className="input-wrapper">
                                    <input
                                        type="text"
                                        id="username"
                                        name="username"
                                        placeholder="Nhập tên đăng nhập hoặc email"
                                        required
                                        value={formData.username}
                                        onChange={handleChange}
                                    />
                                </div>
                            </div>

                            <div className="input-group-modern">
                                <label htmlFor="password">Mật khẩu</label>
                                <div className="input-wrapper">
                                    <input
                                        type={showPassword ? "text" : "password"}
                                        id="password"
                                        name="password"
                                        placeholder="••••••••"
                                        required
                                        value={formData.password}
                                        onChange={handleChange}
                                    />
                                    <button
                                        type="button"
                                        className="password-toggle-icon"
                                        onClick={togglePasswordVisibility}
                                    >
                                        {showPassword ? (
                                            <i className="bi bi-eye-slash"></i>
                                        ) : (
                                            <i className="bi bi-eye"></i>
                                        )}
                                    </button>
                                </div>
                            </div>

                            {status.message && (
                                <div className={`status-message ${status.type}`}>
                                    {status.message}
                                </div>
                            )}

                            <button type="submit" className="btn-modern-submit" disabled={status.loading}>
                                {status.loading ? 'Đang xử lý...' : 'Đăng nhập'}
                            </button>
                        </form>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Login;
