import { useState, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';

/* ── Features grid data ────────────────────────────────────────────────── */
const features = [
    { icon: '🎯', title: 'Player Tracking', desc: 'AI-powered player detection and tracking from match video' },
    { icon: '📊', title: 'Advanced Analytics', desc: 'xG, xT, VAEP — professional-grade football metrics' },
    { icon: '🧠', title: 'Style Embeddings', desc: 'Deep learning player style analysis with clustering' },
    { icon: '⭐', title: 'Player Ratings', desc: 'Multi-factor player rating on a 0-10 scale' },
    { icon: '🗺️', title: 'Heatmaps', desc: 'Visual representation of player positioning and movement' },
    { icon: '🎙️', title: 'AI Commentary', desc: 'Auto-generated expert commentary for your match' },
];

/* ── Capabilities slideshow data ───────────────────────────────────────── */
const capabilities = [
    {
        title: 'End-to-End Video Analysis',
        desc: 'Upload a raw match video and receive a complete analytical breakdown — no manual annotation, no spreadsheets, just intelligent automation from start to finish.',
        icon: '🎬',
        color: '#6366f1',
    },
    {
        title: 'Professional-Grade Metrics',
        desc: 'Your matches analyzed with the same advanced metrics used by top clubs: Expected Goals (xG), Expected Threat (xT), and VAEP — all calibrated for grassroots football.',
        icon: '📈',
        color: '#8b5cf6',
    },
    {
        title: 'Player Style Intelligence',
        desc: 'Our deep learning pipeline learns a unique style fingerprint for every player using 26 features, then clusters similar players together on an interactive style map.',
        icon: '🧬',
        color: '#a855f7',
    },
    {
        title: 'AI-Generated Commentary',
        desc: 'Two specialized AI components merge expert tactical analysis with play-by-play narration into a single, professional commentary video — covering the full match from whistle to whistle.',
        icon: '🎙️',
        color: '#ec4899',
    },
    {
        title: 'Compare & Discover',
        desc: 'Overlay radar charts, side-by-side stat diffs, heatmap comparisons — find hidden gems and benchmark players against teammates or opponents in seconds.',
        icon: '🔍',
        color: '#22c55e',
    },
    {
        title: 'GPU-Accelerated, Cloud-Ready',
        desc: 'Designed for vast.ai and runpod GPU instances. Auto-detects CUDA, MPS, or CPU — deploy anywhere from a local laptop to a rented cloud GPU in minutes.',
        icon: '🚀',
        color: '#f59e0b',
    },
];

/* ── Slideshow hook ────────────────────────────────────────────────────── */
function useSlideshow(total: number, intervalMs = 5000) {
    const [current, setCurrent] = useState(0);
    const timerRef = useRef<ReturnType<typeof setInterval>>(undefined);

    const reset = () => {
        if (timerRef.current) clearInterval(timerRef.current);
        timerRef.current = setInterval(() => setCurrent(p => (p + 1) % total), intervalMs);
    };

    useEffect(() => { reset(); return () => clearInterval(timerRef.current); }, [total]);

    const goTo = (idx: number) => { setCurrent(idx); reset(); };
    const next = () => goTo((current + 1) % total);
    const prev = () => goTo((current - 1 + total) % total);

    return { current, goTo, next, prev };
}

/* ── Landing page ──────────────────────────────────────────────────────── */
export default function Landing() {
    const slideshow = useSlideshow(capabilities.length, 5000);
    const cap = capabilities[slideshow.current];

    return (
        <div>
            {/* ▸ Hero Section */}
            <section style={{
                background: 'var(--gradient-hero)',
                minHeight: '85vh',
                display: 'flex',
                alignItems: 'center',
                position: 'relative',
                overflow: 'hidden',
                transition: 'background 0.4s ease',
            }}>
                {/* Ambient orbs */}
                <div style={{
                    position: 'absolute', width: '600px', height: '600px', borderRadius: '50%',
                    background: 'radial-gradient(circle, rgba(99,102,241,0.08) 0%, transparent 70%)',
                    top: '-200px', right: '-100px', pointerEvents: 'none',
                }} />
                <div style={{
                    position: 'absolute', width: '400px', height: '400px', borderRadius: '50%',
                    background: 'radial-gradient(circle, rgba(168,85,247,0.06) 0%, transparent 70%)',
                    bottom: '-100px', left: '-50px', pointerEvents: 'none',
                }} />

                <div className="page-container" style={{ width: '100%', position: 'relative', zIndex: 1 }}>
                    <div style={{ maxWidth: '720px' }}>
                        <div style={{
                            display: 'inline-block', padding: '6px 16px', borderRadius: '20px',
                            background: 'rgba(99,102,241,0.15)', color: 'var(--accent-light)',
                            fontSize: '0.85rem', fontWeight: 600, marginBottom: '24px',
                            border: '1px solid rgba(99,102,241,0.25)',
                        }}>
                            🚀 AI-Powered Analysis &amp; Commentary for Every Match
                        </div>

                        <h1 style={{ fontSize: '3.5rem', fontWeight: 900, lineHeight: 1.1, marginBottom: '20px' }}>
                            <span style={{
                                background: 'var(--gradient-primary)',
                                WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
                            }}>Professional Football</span>
                            <br />
                            <span style={{ color: 'var(--text-primary)' }}>Analytics &amp; Commentary</span>
                        </h1>

                        <p style={{
                            fontSize: '1.2rem', color: 'var(--text-secondary)', lineHeight: 1.7,
                            marginBottom: '24px', maxWidth: '580px',
                        }}>
                            Upload your match video and get instant access to player tracking, advanced metrics,
                            AI-generated expert commentary, and professional-grade analysis — all powered by cutting-edge AI.
                        </p>

                        {/* Commentary call-out */}
                        <div style={{
                            display: 'flex', alignItems: 'center', gap: '12px',
                            padding: '14px 20px', borderRadius: '14px',
                            background: 'rgba(236, 72, 153, 0.08)', border: '1px solid rgba(236, 72, 153, 0.2)',
                            marginBottom: '32px', maxWidth: '540px',
                        }}>
                            <span style={{ fontSize: '1.6rem' }}>🎙️</span>
                            <div>
                                <div style={{ fontWeight: 700, fontSize: '0.95rem', color: '#ec4899' }}>
                                    AI Commentary Included
                                </div>
                                <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                                    Expert tactical analysis + play-by-play narration — merged into one professional commentary video
                                </div>
                            </div>
                        </div>

                        <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
                            <Link to="/upload">
                                <button className="btn-primary" style={{ fontSize: '1.1rem', padding: '16px 40px' }}>
                                    Upload Match Video →
                                </button>
                            </Link>
                            <button className="btn-secondary">Watch Demo</button>
                        </div>
                    </div>
                </div>
            </section>

            {/* ▸ Features Grid */}
            <section className="page-container" style={{ paddingTop: '80px', paddingBottom: '48px' }}>
                <div style={{ textAlign: 'center', marginBottom: '48px' }}>
                    <h2 className="page-title" style={{ fontSize: '2.5rem' }}>What We Analyze</h2>
                    <p className="page-subtitle">From raw video to professional-grade insights in minutes</p>
                </div>

                <div style={{
                    display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
                    gap: '24px',
                }}>
                    {features.map((f, i) => (
                        <div
                            key={i}
                            className="glass-card animate-fade-in-up"
                            style={{ animationDelay: `${i * 0.1}s`, opacity: 0 }}
                        >
                            <div style={{ fontSize: '2.5rem', marginBottom: '16px' }}>{f.icon}</div>
                            <h3 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '8px' }}>{f.title}</h3>
                            <p style={{ color: 'var(--text-secondary)' }}>{f.desc}</p>
                        </div>
                    ))}
                </div>
            </section>

            {/* ▸ Capabilities Slideshow */}
            <section style={{
                background: 'var(--bg-secondary)', borderTop: '1px solid var(--border-subtle)',
                borderBottom: '1px solid var(--border-subtle)', transition: 'background 0.4s ease',
            }}>
                <div className="page-container" style={{ paddingTop: '72px', paddingBottom: '72px' }}>
                    <div style={{ textAlign: 'center', marginBottom: '48px' }}>
                        <h2 className="page-title" style={{ fontSize: '2.2rem' }}>Platform Capabilities</h2>
                        <p className="page-subtitle" style={{ marginBottom: 0 }}>Everything your match video unlocks</p>
                    </div>

                    {/* Slide content */}
                    <div className="glass-card" style={{
                        maxWidth: '800px', margin: '0 auto', padding: '48px 40px',
                        textAlign: 'center', minHeight: '260px',
                        display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                    }}>
                        <div style={{
                            width: '72px', height: '72px', borderRadius: '50%',
                            background: `${cap.color}18`, border: `2px solid ${cap.color}44`,
                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                            fontSize: '2rem', marginBottom: '20px',
                            transition: 'all 0.4s ease',
                        }}>
                            {cap.icon}
                        </div>
                        <h3 style={{
                            fontSize: '1.5rem', fontWeight: 800, marginBottom: '12px',
                            color: cap.color, transition: 'color 0.4s ease',
                        }}>
                            {cap.title}
                        </h3>
                        <p style={{
                            color: 'var(--text-secondary)', fontSize: '1.05rem',
                            lineHeight: 1.7, maxWidth: '600px',
                        }}>
                            {cap.desc}
                        </p>
                    </div>

                    {/* Slideshow controls */}
                    <div style={{
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        gap: '16px', marginTop: '28px',
                    }}>
                        <button onClick={slideshow.prev} className="btn-secondary"
                            style={{ width: '40px', height: '40px', padding: 0, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                            ‹
                        </button>

                        <div style={{ display: 'flex', gap: '8px' }}>
                            {capabilities.map((_, i) => (
                                <button
                                    key={i}
                                    onClick={() => slideshow.goTo(i)}
                                    style={{
                                        width: slideshow.current === i ? '28px' : '10px',
                                        height: '10px',
                                        borderRadius: '5px',
                                        background: slideshow.current === i ? 'var(--accent)' : 'var(--border-subtle)',
                                        border: 'none',
                                        cursor: 'pointer',
                                        transition: 'all 0.3s ease',
                                    }}
                                />
                            ))}
                        </div>

                        <button onClick={slideshow.next} className="btn-secondary"
                            style={{ width: '40px', height: '40px', padding: 0, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                            ›
                        </button>
                    </div>
                </div>
            </section>

            {/* ▸ Commentary Highlight */}
            <section className="page-container" style={{ paddingTop: '72px', paddingBottom: '72px' }}>
                <div style={{
                    display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '48px', alignItems: 'center',
                }}>
                    <div>
                        <div style={{
                            display: 'inline-block', padding: '5px 14px', borderRadius: '20px',
                            background: 'rgba(236, 72, 153, 0.12)', color: '#ec4899',
                            fontSize: '0.8rem', fontWeight: 700, marginBottom: '16px',
                            border: '1px solid rgba(236, 72, 153, 0.25)', textTransform: 'uppercase',
                            letterSpacing: '0.06em',
                        }}>
                            ★ Core Feature
                        </div>
                        <h2 style={{ fontSize: '2rem', fontWeight: 800, marginBottom: '16px', color: 'var(--text-primary)' }}>
                            AI-Generated <span style={{ color: '#ec4899' }}>Commentary</span>
                        </h2>
                        <p style={{ color: 'var(--text-secondary)', lineHeight: 1.8, marginBottom: '20px', fontSize: '1.05rem' }}>
                            Our commentary system combines two independent AI components into a single, seamless video:
                        </p>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                            {[
                                { label: 'Expert Analysis', desc: 'Tactical insights powered by xG, xT, VAEP, and team patterns', icon: '📊' },
                                { label: 'Match Narration', desc: 'Play-by-play coverage of every key moment', icon: '📢' },
                                { label: 'Single Video Output', desc: 'Both commentary tracks merged into one cohesive final production', icon: '🎞️' },
                            ].map((item, i) => (
                                <div key={i} style={{
                                    display: 'flex', gap: '14px', padding: '14px 18px',
                                    borderRadius: '12px', background: 'var(--bg-card)', border: '1px solid var(--border-subtle)',
                                    transition: 'all 0.3s',
                                }}>
                                    <span style={{ fontSize: '1.4rem', flexShrink: 0 }}>{item.icon}</span>
                                    <div>
                                        <div style={{ fontWeight: 700, fontSize: '0.95rem' }}>{item.label}</div>
                                        <div style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>{item.desc}</div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Commentary visual */}
                    <div className="glass-card" style={{
                        aspectRatio: '16/10', display: 'flex', flexDirection: 'column',
                        alignItems: 'center', justifyContent: 'center', textAlign: 'center',
                    }}>
                        <div style={{ fontSize: '4rem', marginBottom: '16px' }}>🎙️</div>
                        <h3 style={{ fontWeight: 700, marginBottom: '8px', fontSize: '1.3rem' }}>Full Match Commentary</h3>
                        <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem', maxWidth: '280px' }}>
                            Automatically generated the moment your video finishes processing
                        </p>
                        <div style={{
                            marginTop: '20px', display: 'flex', gap: '8px',
                        }}>
                            {['Expert Analysis', 'General'].map(tag => (
                                <span key={tag} style={{
                                    padding: '5px 12px', borderRadius: '20px', fontSize: '0.75rem', fontWeight: 600,
                                    background: 'rgba(236, 72, 153, 0.12)', color: '#ec4899',
                                    border: '1px solid rgba(236, 72, 153, 0.25)',
                                }}>
                                    {tag}
                                </span>
                            ))}
                        </div>
                    </div>
                </div>
            </section>

            {/* ▸ CTA */}
            <section style={{
                background: 'var(--bg-card)',
                padding: '80px 24px',
                textAlign: 'center',
                borderTop: '1px solid var(--border-subtle)',
                transition: 'background 0.4s ease',
            }}>
                <h2 style={{
                    fontSize: '2rem', fontWeight: 700, marginBottom: '16px',
                    background: 'var(--gradient-primary)',
                    WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
                }}>
                    Ready to Analyze Your Match?
                </h2>
                <p style={{ color: 'var(--text-secondary)', marginBottom: '32px', fontSize: '1.1rem' }}>
                    Upload your video and our AI pipeline handles everything — analytics, ratings, and commentary.
                </p>
                <Link to="/upload">
                    <button className="btn-primary" style={{ fontSize: '1.1rem', padding: '16px 48px' }}>
                        Get Started Free →
                    </button>
                </Link>
            </section>
        </div>
    );
}
