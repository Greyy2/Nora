import React, { createContext, useContext, useState, useEffect } from 'react';

const ThemeContext = createContext();

export const useTheme = () => useContext(ThemeContext);

export const ThemeProvider = ({ children }) => {
    const [theme, setTheme] = useState(() => {
        try {
            const storedTheme = localStorage.getItem('nora_theme');
            return storedTheme === 'light' ? 'light' : 'dark';
        } catch {
            return 'dark';
        }
    });

    useEffect(() => {
        const normalizedTheme = theme === 'light' ? 'light' : 'dark';

        document.body.classList.remove('light', 'dark');
        document.body.classList.add(normalizedTheme);

        document.documentElement.setAttribute('data-theme', normalizedTheme);
        document.documentElement.classList.toggle('dark', normalizedTheme === 'dark');
        document.documentElement.classList.toggle('light', normalizedTheme === 'light');

        try {
            localStorage.setItem('nora_theme', normalizedTheme);
        } catch {
            // Ignore storage errors (private mode / disabled storage)
        }
    }, [theme]);

    const toggleTheme = () => {
        setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'));
    };

    return (
        <ThemeContext value={{ theme, toggleTheme }}>
            {children}
        </ThemeContext>
    );
};
