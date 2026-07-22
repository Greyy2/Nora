import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../user/AuthContext';

/**
 * Custom Hook for Login Logic
 * Handles state, form submission, and redirection
 */
export const useLoginLogic = () => {
    const navigate = useNavigate();
    const { login, isAuthenticated } = useAuth();
    const [formData, setFormData] = useState({
        username: '',
        password: '',
        remember: false
    });
    const [status, setStatus] = useState({
        loading: false,
        message: '',
        type: '' // 'success' | 'error'
    });
    const [showPassword, setShowPassword] = useState(false);

    // Removed auto-redirect - let user navigate manually

    const handleChange = (e) => {
        const { name, value, type, checked } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: type === 'checkbox' ? checked : value
        }));
    };

    const togglePasswordVisibility = () => {
        setShowPassword(!showPassword);
    };

    const handleLogin = async (e) => {
        e.preventDefault();

        const { username } = formData;

        if (!username || username.trim().length < 2) {
            setStatus({ loading: false, message: 'Please enter your name or email', type: 'error' });
            return;
        }

        setStatus({ loading: true, message: '⏳ Logging in...', type: 'info' });

        // Simulate login delay
        await new Promise(resolve => setTimeout(resolve, 300));

        // DEMO LOGIN LOGIC
        const displayName = username.includes('@') ? username.split('@')[0] : username;
        const userData = {
            username: displayName,
            email: username.includes('@') ? username : `${username}@demo.com`,
            loginTime: new Date().toISOString(),
            loginMethod: 'form'
        };

        login(userData);

        setStatus({ loading: false, message: `Welcome ${displayName}!`, type: 'success' });

        setTimeout(() => {
            sessionStorage.setItem('loginSuccess', 'Đăng nhập thành công! Chào mừng ' + displayName + '.');
            const redirectUrl = localStorage.getItem('redirectAfterLogin') || '/home';
            localStorage.removeItem('redirectAfterLogin');
            navigate(redirectUrl);
        }, 500);
    };

    const handleGoogleLogin = async () => {
        const name = prompt('Enter your name (Demo Google Login):');
        if (name && name.trim().length >= 2) {
            setStatus({ loading: true, message: '⏳ Signing in...', type: 'info' });

            await new Promise(resolve => setTimeout(resolve, 500));

            const displayName = name.trim();
            const userData = {
                username: displayName,
                email: `${displayName.toLowerCase().replace(/\s+/g, '.')}@gmail.com`,
                loginTime: new Date().toISOString(),
                loginMethod: 'google'
            };

            login(userData);

            setStatus({ loading: false, message: `Welcome ${displayName}!`, type: 'success' });

            setTimeout(() => {
                sessionStorage.setItem('loginSuccess', 'Đăng nhập thành công! Chào mừng ' + displayName + '.');
                const redirectUrl = localStorage.getItem('redirectAfterLogin') || '/home';
                localStorage.removeItem('redirectAfterLogin');
                navigate(redirectUrl);
            }, 500);
        }
    };

    return {
        formData,
        status,
        showPassword,
        handleChange,
        togglePasswordVisibility,
        handleLogin,
        handleGoogleLogin
    };
};
