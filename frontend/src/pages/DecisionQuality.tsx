import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { getDecisionQuality } from '../api/client';

interface PlayerDQ {
    player_id: number | null;
    player_name: string;
    team: string;
    dq_score: number;
    tier: string;
    weighted_dq: number;
    n_actions: number;
    pct_optimal: number;
}

interface Decision {
    event_uuid: string;
    player_name: string;
    team: string;
    minute: number;
    second: number;
    period: number;
    event_type: string;
    action_type: string;
    dq_score: number;
    description: string;
}

interface DQData {
    players: PlayerDQ[];
    best_decisions: Decision[];
    worst_decisions: Decision[];
    total_events_analyzed: number;
}

const TIER_COLORS: Record<string, { bg: string; text: string; border: string }> = {
    Elite: { bg: 'rgba(234,179,8,0.15)', text: '#eab308', border: 'rgba(234,179,8,0.4)' },
    'Very Good': { bg: 'rgba(34,197,94,0.15)', text: '#22c55e', border: 'rgba(34,197,94,0.4)' },
    Good: { bg: 'rgba(99,102,241,0.15)', text: '#818cf8', border: 'rgba(99,102,241,0.4)' },
    Average: { bg: 'rgba(148,163,184,0.1)', text: '#94a3b8', border: 'rgba(148,163,184,0.3)' },
    'Below Average': { bg: 'rgba(249,115,22,0.12)', text: '#f97316', border: 'rgba(249,115,22,0.3)' },
    Poor: { bg: 'rgba(239,68,68,0.12)', text: '#ef4444', border: 'rgba(239,68,68,0.3)' },
};

function ScoreBar({ score, tier }: { score: number; tier: string }) {
    const colors = TIER_COLORS[tier] || TIER_COLORS['Average'];
    const clampedScore = Math.max(0, Math.min(100, score));
    return (
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', minWidth: '180px' }}>
            <div style={{
                flex: 1, height: '8px', borderRadius: '4px',
                background: 'var(--bg-secondary)', overflow: 'hidden',
            }}>
                <div style={{
                    width: `${clampedScore}%`,
                    height: '100%',
                    borderRadius: '4px',
                    background: colors.text,
                    transition: 'width 0.6s ease',
                }} />
            </div>
            <span style={{ fontWeight: 700, fontSize: '0.95rem', minWidth: '36px', textAlign: 'right', color: colors.text }}>
                {score.toFixed(0)}
            </span>
        </div>
    );
}

function TierBadge({ tier }: { tier: string }) {
    const colors = TIER_COLORS[tier] || TIER_COLORS['Average'];
    return (
        <span style={{
            padding: '3px 10px', borderRadius: '12px', fontSize: '0.75rem', fontWeight: 700,
            background: colors.bg, color: colors.text,
            border: `1px solid ${colors.border}`,
            letterSpacing: '0.02em',
        }}>
            {tier}
        </span>
    );
}

function DecisionRow({ d, isBest }: { d: Decision; isBest: boolean }) {
    const accentColor = isBest ? '#22c55e' : '#ef4444';
    const label = isBest ? '▲' : '▼';
    return (
        <div style={{
            display: 'grid',
            gridTemplateColumns: '28px 1fr auto auto auto',
            gap: '12px',
            padding: '12px 16px',
            borderBottom: '1px solid var(--border-subtle)',
            alignItems: 'center',
        }}>
            <span style={{ color: accentColor, fontWeight: 700, fontSize: '0.9rem' }}>{label}</span>
            <div>
                <div style={{ fontWeight: 600, fontSize: '0.9rem' }}>{d.player_name}</div>
                <div style={{ color: 'var(--text-muted)', fontSize: '0.78rem' }}>{d.description}</div>
            </div>
            <span style={{
                padding: '3px 8px', borderRadius: '6px', fontSize: '0.75rem', fontWeight: 600,
                background: d.event_type === 'Shot' ? 'rgba(239,68,68,0.15)'
                    : d.event_type === 'Pass' ? 'rgba(99,102,241,0.15)'
                        : 'rgba(148,163,184,0.1)',
                color: d.event_type === 'Shot' ? '#ef4444'
                    : d.event_type === 'Pass' ? '#818cf8'
                        : 'var(--text-secondary)',
            }}>
                {d.action_type}
            </span>
            <span style={{ color: 'var(--text-muted)', fontSize: '0.82rem', whiteSpace: 'nowrap' }}>
                P{d.period} {String(d.minute).padStart(2, '0')}:{String(d.second).padStart(2, '0')}
            </span>
            <span style={{
                fontWeight: 700, fontSize: '0.9rem',
                color: isBest ? '#22c55e' : '#ef4444',
                textAlign: 'right', minWidth: '60px',
            }}>
                {d.dq_score.toFixed(3)}
            </span>
        </div>
    );
}

export default function DecisionQuality() {
    const { id } = useParams();
    const matchId = Number(id);
    const [data, setData] = useState<DQData | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [activeTeam, setActiveTeam] = useState<string>('All');

    useEffect(() => {
        setLoading(true);
        getDecisionQuality(matchId)
            .then(r => { setData(r.data); setLoading(false); })
            .catch(e => {
                setError(e?.response?.data?.detail || 'Failed to load Decision Quality data');
                setLoading(false);
            });
    }, [matchId]);

    if (loading) return (
        <div className="page-container">
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px', paddingTop: '80px' }}>
                <div className="spinner" />
                <p style={{ color: 'var(--text-secondary)' }}>Computing decision quality…</p>
            </div>
        </div>
    );

    if (error) return (
        <div className="page-container">
            <div className="glass-card" style={{ textAlign: 'center', padding: '48px' }}>
                <p style={{ color: 'var(--danger)', fontSize: '1rem' }}>{error}</p>
                <Link to={`/match/${matchId}`}>
                    <button className="btn-secondary" style={{ marginTop: '16px' }}>← Back to Match</button>
                </Link>
            </div>
        </div>
    );

    if (!data) return null;

    const teams = ['All', ...Array.from(new Set(data.players.map(p => p.team).filter(Boolean)))];
    const filteredPlayers = activeTeam === 'All'
        ? data.players
        : data.players.filter(p => p.team === activeTeam);

    const tierCounts = data.players.reduce<Record<string, number>>((acc, p) => {
        acc[p.tier] = (acc[p.tier] || 0) + 1;
        return acc;
    }, {});

    return (
        <div className="page-container">
            {/* Header */}
            <div style={{ marginBottom: '32px' }}>
                <Link to={`/match/${matchId}`} style={{ color: 'var(--text-muted)', fontSize: '0.85rem', textDecoration: 'none' }}>
                    ← Back to Match
                </Link>
                <h1 style={{ fontSize: '2rem', fontWeight: 800, marginTop: '8px' }}>
                    Decision Quality
                </h1>
                <p style={{ color: 'var(--text-secondary)', marginTop: '4px' }}>
                    How well did each player choose their actions given the game situation?
                </p>
            </div>

            {/* Summary row */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '16px', marginBottom: '32px' }}>
                {[
                    { label: 'Events Analysed', value: data.total_events_analyzed },
                    { label: 'Players Scored', value: data.players.length },
                    { label: 'Avg DQ Score', value: data.players.length ? (data.players.reduce((s, p) => s + p.dq_score, 0) / data.players.length).toFixed(1) : '—' },
                    { label: 'Top Scorer', value: data.players[0]?.player_name.split(' ').slice(-1)[0] || '—' },
                ].map((s, i) => (
                    <div key={i} className="glass-card" style={{ textAlign: 'center', padding: '20px 12px' }}>
                        <div style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--accent-light)' }}>{s.value}</div>
                        <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginTop: '4px' }}>{s.label}</div>
                    </div>
                ))}
            </div>

            {/* Best / Worst decisions */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', marginBottom: '32px' }}>
                {/* Best 3 */}
                <div className="glass-card" style={{ padding: '0' }}>
                    <div style={{ padding: '16px 16px 12px', borderBottom: '1px solid var(--border-subtle)' }}>
                        <h3 style={{ fontSize: '1rem', fontWeight: 700 }}>
                            <span style={{ color: '#22c55e', marginRight: '8px' }}>▲</span>
                            Best 3 Decisions
                        </h3>
                        <p style={{ color: 'var(--text-muted)', fontSize: '0.78rem', marginTop: '2px' }}>
                            Highest Decision Quality score in the match
                        </p>
                    </div>
                    {data.best_decisions.length > 0 ? (
                        data.best_decisions.map((d, i) => (
                            <DecisionRow key={i} d={d} isBest={true} />
                        ))
                    ) : (
                        <div style={{ padding: '24px 16px', color: 'var(--text-muted)', textAlign: 'center', fontSize: '0.9rem' }}>
                            Not enough data
                        </div>
                    )}
                </div>

                {/* Worst 3 */}
                <div className="glass-card" style={{ padding: '0' }}>
                    <div style={{ padding: '16px 16px 12px', borderBottom: '1px solid var(--border-subtle)' }}>
                        <h3 style={{ fontSize: '1rem', fontWeight: 700 }}>
                            <span style={{ color: '#ef4444', marginRight: '8px' }}>▼</span>
                            Worst 3 Decisions
                        </h3>
                        <p style={{ color: 'var(--text-muted)', fontSize: '0.78rem', marginTop: '2px' }}>
                            Lowest Decision Quality score in the match
                        </p>
                    </div>
                    {data.worst_decisions.length > 0 ? (
                        [...data.worst_decisions].reverse().map((d, i) => (
                            <DecisionRow key={i} d={d} isBest={false} />
                        ))
                    ) : (
                        <div style={{ padding: '24px 16px', color: 'var(--text-muted)', textAlign: 'center', fontSize: '0.9rem' }}>
                            Not enough data
                        </div>
                    )}
                </div>
            </div>

            {/* Score interpretation legend */}
            <div className="glass-card" style={{ marginBottom: '24px', padding: '16px 20px' }}>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px', alignItems: 'center' }}>
                    <span style={{ color: 'var(--text-secondary)', fontSize: '0.82rem', fontWeight: 600, marginRight: '4px' }}>
                        Score guide:
                    </span>
                    {Object.entries(TIER_COLORS).map(([tier, col]) => (
                        <span key={tier} style={{
                            display: 'flex', alignItems: 'center', gap: '6px',
                            fontSize: '0.78rem', color: col.text,
                        }}>
                            <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: col.text, display: 'inline-block' }} />
                            <strong>{tier}</strong>
                            <span style={{ color: 'var(--text-muted)' }}>
                                {tier === 'Elite' ? '83+' : tier === 'Very Good' ? '69–82' : tier === 'Good' ? '55–68' : tier === 'Average' ? '45–54' : tier === 'Below Average' ? '31–44' : '0–30'}
                            </span>
                        </span>
                    ))}
                    <span style={{ marginLeft: 'auto', color: 'var(--text-muted)', fontSize: '0.78rem' }}>
                        50 = average for position
                    </span>
                </div>
            </div>

            {/* Team filter */}
            {teams.length > 2 && (
                <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
                    {teams.map(t => (
                        <button
                            key={t}
                            onClick={() => setActiveTeam(t)}
                            style={{
                                padding: '6px 16px', borderRadius: '20px', border: 'none', cursor: 'pointer',
                                fontSize: '0.82rem', fontWeight: 600,
                                background: activeTeam === t ? 'var(--accent)' : 'var(--bg-secondary)',
                                color: activeTeam === t ? '#fff' : 'var(--text-secondary)',
                                transition: 'all 0.2s ease',
                            }}
                        >
                            {t}
                        </button>
                    ))}
                </div>
            )}

            {/* Player DQ table */}
            <div className="glass-card" style={{ padding: '0', marginBottom: '32px' }}>
                <div style={{ padding: '16px 20px 12px', borderBottom: '1px solid var(--border-subtle)' }}>
                    <h3 style={{ fontSize: '1rem', fontWeight: 700 }}>Player Decision Quality Scores</h3>
                    <p style={{ color: 'var(--text-muted)', fontSize: '0.78rem', marginTop: '2px' }}>
                        Ranked by weighted DQ score · 50 = average · based on {data.total_events_analyzed} events
                    </p>
                </div>

                <div style={{ overflowX: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                        <thead>
                            <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                                {['Rank', 'Player', 'Team', 'Score', 'Tier', 'Actions', '% Optimal'].map(h => (
                                    <th key={h} style={{
                                        padding: '12px 16px', textAlign: 'left',
                                        color: 'var(--text-muted)', fontSize: '0.78rem', fontWeight: 600,
                                        whiteSpace: 'nowrap',
                                    }}>{h}</th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            {filteredPlayers.map((p, i) => (
                                <tr
                                    key={p.player_id ?? p.player_name}
                                    style={{
                                        borderBottom: '1px solid var(--border-subtle)',
                                        background: i % 2 === 0 ? 'transparent' : 'var(--table-stripe)',
                                        transition: 'background 0.15s',
                                    }}
                                    onMouseEnter={e => (e.currentTarget.style.background = 'var(--bg-card-hover)')}
                                    onMouseLeave={e => (e.currentTarget.style.background = i % 2 === 0 ? 'transparent' : 'var(--table-stripe)')}
                                >
                                    <td style={{ padding: '12px 16px', color: 'var(--text-muted)', fontSize: '0.85rem', fontWeight: 600 }}>
                                        {i === 0 && <span style={{ marginRight: '4px' }}>🥇</span>}
                                        {i === 1 && <span style={{ marginRight: '4px' }}>🥈</span>}
                                        {i === 2 && <span style={{ marginRight: '4px' }}>🥉</span>}
                                        #{i + 1}
                                    </td>
                                    <td style={{ padding: '12px 16px', fontWeight: 600 }}>{p.player_name}</td>
                                    <td style={{ padding: '12px 16px', color: 'var(--text-secondary)', fontSize: '0.88rem' }}>{p.team || '—'}</td>
                                    <td style={{ padding: '12px 16px', minWidth: '200px' }}>
                                        <ScoreBar score={p.dq_score} tier={p.tier} />
                                    </td>
                                    <td style={{ padding: '12px 16px' }}>
                                        <TierBadge tier={p.tier} />
                                    </td>
                                    <td style={{ padding: '12px 16px', color: 'var(--text-secondary)', fontSize: '0.88rem' }}>
                                        {p.n_actions}
                                    </td>
                                    <td style={{ padding: '12px 16px', fontSize: '0.88rem' }}>
                                        <span style={{
                                            color: p.pct_optimal >= 15 ? '#22c55e' : p.pct_optimal >= 8 ? '#f59e0b' : '#ef4444',
                                            fontWeight: 600,
                                        }}>
                                            {p.pct_optimal.toFixed(1)}%
                                        </span>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>

                    {filteredPlayers.length === 0 && (
                        <div style={{ padding: '32px', textAlign: 'center', color: 'var(--text-muted)' }}>
                            No players found with enough data for DQ scoring.
                        </div>
                    )}
                </div>
            </div>

            {/* Tier distribution */}
            {data.players.length > 0 && (
                <div className="glass-card" style={{ padding: '20px' }}>
                    <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '16px' }}>Tier Distribution</h3>
                    <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
                        {['Elite', 'Very Good', 'Good', 'Average', 'Below Average', 'Poor'].map(tier => {
                            const count = tierCounts[tier] || 0;
                            const pct = data.players.length ? (count / data.players.length) * 100 : 0;
                            const col = TIER_COLORS[tier];
                            return (
                                <div key={tier} style={{ flex: '1', minWidth: '80px', textAlign: 'center' }}>
                                    <div style={{ fontSize: '1.5rem', fontWeight: 800, color: col.text }}>{count}</div>
                                    <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '2px' }}>{tier}</div>
                                    <div style={{
                                        height: '4px', borderRadius: '2px', marginTop: '6px',
                                        background: col.bg, border: `1px solid ${col.border}`,
                                        overflow: 'hidden',
                                    }}>
                                        <div style={{ width: `${pct}%`, height: '100%', background: col.text, borderRadius: '2px' }} />
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}
        </div>
    );
}
