import React from 'react';
import { useNavigate } from 'react-router-dom';

const ProfileCard = ({ user }) => {
    const navigate = useNavigate();

    if (!user) return null;

    const getInitials = (name) => {
        if (!name) return 'U';
        return name.split(' ')
            .map(n => n[0])
            .join('')
            .toUpperCase()
            .slice(0, 2);
    };

    const displayName = user?.name || user?.username || 'User';
    const displayEmail = user?.email || 'Chưa cập nhật email';
    const initials = getInitials(displayName);

    return (
        <div className="profile-dashboard-card fade-in">
            <div className="profile-card-inner">
                <div className="profile-main-info">
                    <div className="profile-avatar-wrapper">
                        <div className="profile-avatar-circle">
                            {initials}
                        </div>
                        <div className="online-badge"></div>
                    </div>
                    
                    <div className="profile-text-content">
                        <div className="profile-name-row">
                            <h3>{displayName}</h3>
                            <span className="account-tier-badge">Elite Tier</span>
                        </div>
                        <p className="profile-email-text">{displayEmail}</p>
                        <div className="profile-verification">
                            <i className="bi bi-patch-check-fill"></i>
                            <span>Tài khoản đã xác thực</span>
                        </div>
                    </div>
                </div>

                <div className="profile-stats-divider"></div>

                <div className="profile-mini-stats">
                    <div className="mini-stat">
                        <span className="stat-num">42</span>
                        <span className="stat-desc">Chiến lược</span>
                    </div>
                    <div className="mini-stat">
                        <span className="stat-num">86%</span>
                        <span className="stat-desc">Tỉ lệ thắng</span>
                    </div>
                    <div className="mini-stat">
                        <span className="stat-num">Top 3</span>
                        <span className="stat-desc">Xếp hạng</span>
                    </div>
                </div>

                <div className="profile-card-actions">
                    <button className="p-btn p-btn-outline" onClick={() => navigate('/settings')}>
                        <i className="bi bi-gear"></i>
                        Cài đặt
                    </button>
                    <button className="p-btn p-btn-primary" onClick={() => navigate('/trading')}>
                        <i className="bi bi-rocket-takeoff"></i>
                        Vào Dashboard
                    </button>
                </div>
            </div>
        </div>
    );
};

export default ProfileCard;
