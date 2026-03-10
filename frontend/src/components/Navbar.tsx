import { Link, useLocation } from 'react-router-dom';
import ThemeToggle from './ThemeToggle';

export default function Navbar() {
    const location = useLocation();
    const isActive = (path: string) => location.pathname === path ? 'active' : '';

    return (
        <nav className="navbar">
            <div className="navbar-inner">
                <Link to="/" className="nav-logo">
                    <span>⚽</span>
                    <span>FootballAI</span>
                </Link>
                <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
                    <div className="nav-links">
                        <Link to="/" className={isActive('/')}>Home</Link>
                        <Link to="/upload" className={isActive('/upload')}>Upload</Link>
                    </div>
                    <ThemeToggle />
                </div>
            </div>
        </nav>
    );
}
