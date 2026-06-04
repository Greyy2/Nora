import React from 'react';

const BrandIcon = ({ size = 20, className = '' }) => {
    const pixelSize = typeof size === 'number' ? `${size}px` : size;

    return (
        <span
            className={className}
            aria-hidden="true"
            style={{
                display: 'inline-flex',
                width: pixelSize,
                height: pixelSize,
                flexShrink: 0,
                lineHeight: 0,
            }}
        >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width="100%" height="100%" role="img">
                <rect width="32" height="32" rx="8" fill="#123a78" />
                <path
                    d="M7 21 L13 15 L17 18 L25 10"
                    fill="none"
                    stroke="#38bdf8"
                    strokeWidth="2.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                />
                <circle cx="25" cy="10" r="2" fill="#f8fafc" />
            </svg>
        </span>
    );
};

export default BrandIcon;
