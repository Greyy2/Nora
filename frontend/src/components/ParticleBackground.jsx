import React, { useEffect, useRef } from 'react';
import { useTheme } from '../user/ThemeContext';

const ParticleBackground = ({
    densityMultiplier = 1,
    minParticles = 40,
    maxParticles = 100,
    connectionDistance = 150,
    speedMultiplier = 1,
    zIndex = 1,
    ambientGlow = true,
    enableConnections = true,
    connectionOpacity = 0.25
}) => {
    const canvasRef = useRef(null);
    const animationRef = useRef(null);
    const particlesRef = useRef(null); // Keep particles persistent
    const { theme } = useTheme();

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const prefersReducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches;
        if (prefersReducedMotion) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Cancel previous animation if exists
        if (animationRef.current) {
            cancelAnimationFrame(animationRef.current);
        }

        // Set canvas size
        const resizeCanvas = () => {
            const root = document.getElementById('root');
            const oldWidth = canvas.width;
            const oldHeight = canvas.height;

            if (root) {
                canvas.width = root.offsetWidth;
                canvas.height = root.offsetHeight;
            } else {
                const scale = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--app-scale')) || 1;
                canvas.width = window.innerWidth / scale;
                canvas.height = window.innerHeight / scale;
            }

            // SMOOTH RESCALE instead of clearing
            if (particlesRef.current && oldWidth > 0 && oldHeight > 0) {
                const ratioX = canvas.width / oldWidth;
                const ratioY = canvas.height / oldHeight;
                particlesRef.current.forEach(p => {
                    p.x *= ratioX;
                    p.y *= ratioY;
                });
            } else {
                particlesRef.current = [];
            }
        };
        resizeCanvas();

        // Dynamic density based on screen area (Roughly 1 particle per 25000 pixels)
        const area = canvas.width * canvas.height;
        const baselineCount = Math.floor(area / 25000);
        const particleCount = Math.max(
            minParticles,
            Math.min(maxParticles, Math.floor(baselineCount * densityMultiplier))
        );

        // Color palette - HIGH SATURATION for "WOW" effect
        const colors = theme === 'dark'
            ? [
                [255, 230, 100],   // Vibrant Gold ✨
                [100, 255, 180],   // Toxic Mint
                [180, 150, 255],   // Electric Purple
                [0, 255, 255],     // Pure Cyan
                [255, 100, 200]    // Hot Pink
            ]
            : [
                [40, 30, 220],     // Saturated Royal Blue
                [180, 20, 180],    // Saturated Deep Purple
                [230, 10, 80],     // Saturated Deep Crimson
                [0, 100, 160]      // Saturated Deep Teal
            ];

        // Create particles
        if (!particlesRef.current || particlesRef.current.length === 0) {
            particlesRef.current = [];

            for (let i = 0; i < particleCount; i++) {
                const x = Math.random() * canvas.width;
                const y = Math.random() * canvas.height;

                particlesRef.current.push({
                    x, y,
                    vx: (Math.random() - 0.5) * 0.4 * speedMultiplier, // Linear velocity for free float
                    vy: (Math.random() - 0.5) * 0.4 * speedMultiplier,
                    baseSize: Math.random() * 3.5 + 1.2, // Wider size variance
                    pulsePhase: Math.random() * Math.PI * 2,
                    pulseSpeed: 0.01 + Math.random() * 0.02,
                    baseOpacity: theme === 'dark'
                        ? Math.random() * 0.3 + 0.6
                        : Math.random() * 0.2 + 0.75, // Higher opacity for light mode visibility
                    colorIndex: Math.floor(Math.random() * colors.length),
                    flickerPhase: Math.random() * Math.PI * 2,
                    flickerSpeed: 0.02 + Math.random() * 0.03
                });
            }
        }

        const particles = particlesRef.current;

        const maxDistanceSquared = connectionDistance * connectionDistance;

        const animate = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            particles.forEach((particle, i) => {
                // Linear movement
                particle.x += particle.vx;
                particle.y += particle.vy;

                // Border Wrapping
                if (particle.x < -50) particle.x = canvas.width + 50;
                if (particle.x > canvas.width + 50) particle.x = -50;
                if (particle.y < -50) particle.y = canvas.height + 50;
                if (particle.y > canvas.height + 50) particle.y = -50;

                particle.pulsePhase += particle.pulseSpeed;
                particle.flickerPhase += particle.flickerSpeed;

                // Size pulsing animation
                const pulse = Math.sin(particle.pulsePhase) * 0.3 + 1;
                const currentSize = particle.baseSize * pulse;

                // Twinkling animation logic
                const flicker = Math.sin(particle.flickerPhase) * (theme === 'dark' ? 0.2 : 0.1) + 0.85;

                const color = colors[particle.colorIndex];
                const opacity = (theme === 'dark' ? particle.baseOpacity : particle.baseOpacity * 0.9) * flicker;

                if (theme === 'dark' && ambientGlow) {
                    // Layer 1: Wide Ambient Glow
                    ctx.beginPath();
                    ctx.arc(particle.x, particle.y, currentSize * 5, 0, Math.PI * 2);
                    ctx.fillStyle = `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${opacity * 0.12})`;
                    ctx.fill();

                    // Layer 2: Medium Glow
                    ctx.beginPath();
                    ctx.arc(particle.x, particle.y, currentSize * 2.5, 0, Math.PI * 2);
                    ctx.fillStyle = `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${opacity * 0.25})`;
                    ctx.fill();

                    // Layer 3: Hot Core Glow
                    ctx.beginPath();
                    ctx.arc(particle.x, particle.y, currentSize * 1.5, 0, Math.PI * 2);
                    ctx.fillStyle = `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${opacity * 0.4})`;
                    ctx.fill();
                }

                // Core Particle
                ctx.beginPath();
                ctx.arc(particle.x, particle.y, currentSize, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${opacity})`;
                ctx.fill();

                // Connections
                if (enableConnections) {
                    for (let j = i + 1; j < particles.length; j += 1) {
                        const otherParticle = particles[j];
                        const dx = particle.x - otherParticle.x;
                        const dy = particle.y - otherParticle.y;
                        const distanceSquared = dx * dx + dy * dy;
                        if (distanceSquared >= maxDistanceSquared) continue;

                        const distance = Math.sqrt(distanceSquared);
                        ctx.beginPath();
                        ctx.moveTo(particle.x, particle.y);
                        ctx.lineTo(otherParticle.x, otherParticle.y);
                        const lineOpacity = connectionOpacity * (1 - distance / connectionDistance) * flicker;
                        ctx.strokeStyle = `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${lineOpacity})`;
                        ctx.lineWidth = 0.7;
                        ctx.stroke();
                    }
                }
            });

            animationRef.current = requestAnimationFrame(animate);
        };

        animate();

        const handleResize = () => {
            resizeCanvas();
        };

        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current);
            }
        };
    }, [theme, densityMultiplier, minParticles, maxParticles, connectionDistance, speedMultiplier, ambientGlow, enableConnections, connectionOpacity]);

    return (
        <canvas
            ref={canvasRef}
            style={{
                position: 'fixed',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                pointerEvents: 'none',
                zIndex
            }}
        />
    );
};

export default ParticleBackground;
