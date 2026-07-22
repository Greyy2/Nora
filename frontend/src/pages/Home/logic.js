import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../user/AuthContext';

export const useHomeLogic = () => {
    const navigate = useNavigate();
    const { user, logout } = useAuth();
    const [activeSection, setActiveSection] = useState('home');
    const [isMenuOpen, setIsMenuOpen] = useState(false);
    const [showSystemModal, setShowSystemModal] = useState(false);

    const scrollToSection = (id) => {
        setActiveSection(id);
        const element = document.getElementById(id);
        if (element) {
            element.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    };

    const handleLogout = (e) => {
        e.preventDefault();
        logout();
        navigate('/home');
    };

    const toggleMenu = () => setIsMenuOpen(!isMenuOpen);
    
    const openSystemModal = () => setShowSystemModal(true);
    const closeSystemModal = () => setShowSystemModal(false);

    return {
        user,
        activeSection,
        isMenuOpen,
        showSystemModal,
        scrollToSection,
        handleLogout,
        toggleMenu,
        openSystemModal,
        closeSystemModal,
        navigate
    };
};
