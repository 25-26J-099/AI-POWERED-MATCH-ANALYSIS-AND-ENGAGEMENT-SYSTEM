import { useTheme } from '../hooks/useTheme';

export default function ThemeToggle() {
    const { theme, toggleTheme } = useTheme();

    return (
        <button
            onClick={toggleTheme}
            aria-label="Toggle theme"
            style={{
                background: 'var(--bg-card)',
                border: '1px solid var(--border-subtle)',
                borderRadius: '12px',
                padding: '8px 12px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                fontSize: '1.1rem',
                color: 'var(--text-primary)',
                transition: 'all 0.3s ease',
            }}
        >
            <span style={{
                display: 'inline-block',
                transition: 'transform 0.4s ease',
                transform: theme === 'dark' ? 'rotate(0deg)' : 'rotate(180deg)',
            }}>
                {theme === 'dark' ? '🌙' : '☀️'}
            </span>
        </button>
    );
}
