import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { getMatch, getMatchAnalytics, getMatchEvents, getMatchPlayers } from '../api/client';

export default function MatchDashboard() {
    const { id } = useParams();
    const matchId = Number(id);
    const [match, setMatch] = useState<any>(null);
    const [analytics, setAnalytics] = useState<any>(null);
    const [events, setEvents] = useState<any[]>([]);

    const formatEventTime = (event: any) => {
        const minute = Number(event?.minute ?? 0);
        const second = Number(event?.second ?? 0);
        return `${String(minute).padStart(2, '0')}:${String(second).padStart(2, '0')}`;
    };
    const [players, setPlayers] = useState<any[]>([]);

    useEffect(() => {
        getMatch(matchId).then(r => setMatch(r.data));
        getMatchAnalytics(matchId).then(r => setAnalytics(r.data)).catch(() => { });
        getMatchEvents(matchId).then(r => setEvents(r.data)).catch(() => { });
        getMatchPlayers(matchId).then(r => setPlayers(r.data)).catch(() => { });
    }, [matchId]);

    if (!match) return <div className="page-container"><div className="spinner" style={{ margin: '80px auto' }} /></div>;

    const hs = analytics?.home_team_stats || {};
    const as = analytics?.away_team_stats || {};
    const teamColors = Array.isArray(match.team_colors) ? match.team_colors : [];
    const colorForTeam = (name: string | null | undefined, fallbackIndex: number) =>
        teamColors.find((team: any) => team.team_name === name) || teamColors[fallbackIndex];
    const homeColor = colorForTeam(match.home_team, 0);
    const awayColor = colorForTeam(match.away_team, 1);

    const TeamHeader = ({ name, fallback, color }: { name: string | null, fallback: string, color?: any }) => (
        <div>
            <h2 style={{ fontSize: '1.8rem', fontWeight: 800 }}>{name || fallback}</h2>
            {color && (
                <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', marginTop: '8px', color: 'var(--text-secondary)', fontWeight: 600 }}>
                    <span
                        aria-hidden="true"
                        style={{
                            width: '18px',
                            height: '18px',
                            borderRadius: '50%',
                            background: color.hex,
                            border: '2px solid rgba(255,255,255,0.65)',
                        }}
                    />
                    {color.color_name}
                </div>
            )}
        </div>
    );

    return (
        <div className="page-container">
            {/* Header */}
            <div className="glass-card" style={{ textAlign: 'center', marginBottom: '32px', padding: '40px' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '48px' }}>
                    <TeamHeader name={match.home_team} fallback="Home" color={homeColor} />
                    <div style={{
                        fontSize: '1rem', color: 'var(--text-muted)', fontWeight: 600,
                        padding: '12px 24px', borderRadius: '12px', background: 'var(--bg-secondary)',
                    }}>VS</div>
                    <TeamHeader name={match.away_team} fallback="Away" color={awayColor} />
                </div>
                <div style={{ marginTop: '16px', display: 'flex', gap: '12px', justifyContent: 'center' }}>
                    <Link to={`/match/${matchId}/compare`}><button className="btn-secondary">Compare Players</button></Link>
                    <Link to={`/match/${matchId}/styles`}><button className="btn-secondary">Style Map</button></Link>
                    <Link to={`/match/${matchId}/decision-quality`}><button className="btn-secondary">Decision Quality</button></Link>
                    <Link to={`/match/${matchId}/analysis`}><button className="btn-secondary">AI Analysis</button></Link>
                    <Link to={`/match/${matchId}/commentary`}><button className="btn-secondary">Commentary</button></Link>
                </div>
            </div>

            {/* Team Stats Comparison */}
            {analytics && (
                <div style={{ marginBottom: '32px' }}>
                    <h3 style={{ fontSize: '1.3rem', fontWeight: 700, marginBottom: '16px' }}>Team Comparison</h3>
                    <div className="glass-card">
                        {[
                            { label: 'Expected Goals (xG)', home: hs.total_xg?.toFixed(3), away: as.total_xg?.toFixed(3) },
                            { label: 'Expected Threat (xT)', home: hs.total_xt?.toFixed(3), away: as.total_xt?.toFixed(3) },
                            { label: 'VAEP', home: hs.total_vaep?.toFixed(3), away: as.total_vaep?.toFixed(3) },
                            { label: 'Passes', home: hs.total_passes, away: as.total_passes },
                            { label: 'Pass Accuracy', home: `${hs.avg_pass_accuracy?.toFixed(1)}%`, away: `${as.avg_pass_accuracy?.toFixed(1)}%` },
                            { label: 'Shots', home: hs.total_shots, away: as.total_shots },
                        ].map((row, i) => (
                            <div key={i} style={{
                                display: 'grid', gridTemplateColumns: '1fr auto 1fr', gap: '16px',
                                padding: '14px 0', borderBottom: i < 5 ? '1px solid var(--border-subtle)' : 'none',
                                alignItems: 'center',
                            }}>
                                <div style={{ textAlign: 'right', fontSize: '1.1rem', fontWeight: 700 }}>{row.home}</div>
                                <div style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', textAlign: 'center', minWidth: '140px' }}>{row.label}</div>
                                <div style={{ fontSize: '1.1rem', fontWeight: 700 }}>{row.away}</div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Players Table */}
            <div style={{ marginBottom: '32px' }}>
                <h3 style={{ fontSize: '1.3rem', fontWeight: 700, marginBottom: '16px' }}>Players</h3>
                <div className="glass-card" style={{ overflowX: 'auto', padding: '0' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                        <thead>
                            <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                                {['Player', 'Team', 'Position', 'Rating'].map(h => (
                                    <th key={h} style={{ padding: '14px 16px', textAlign: 'left', color: 'var(--text-secondary)', fontSize: '0.85rem', fontWeight: 600 }}>{h}</th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            {players.sort((a, b) => b.rating - a.rating).map((p: any) => (
                                <tr key={p.player_id} style={{ borderBottom: '1px solid var(--border-subtle)', cursor: 'pointer' }}
                                    onClick={() => window.location.href = `/match/${matchId}/player/${p.player_id}`}>
                                    <td style={{ padding: '12px 16px', fontWeight: 600 }}>{p.name}</td>
                                    <td style={{ padding: '12px 16px', color: 'var(--text-secondary)' }}>{p.team}</td>
                                    <td style={{ padding: '12px 16px', color: 'var(--text-secondary)' }}>{p.position || '-'}</td>
                                    <td style={{ padding: '12px 16px' }}>
                                        <span className={`rating-badge ${p.rating >= 8 ? 'rating-excellent' : p.rating >= 6 ? 'rating-good' : p.rating >= 4 ? 'rating-average' : 'rating-poor'}`}
                                            style={{ width: '44px', height: '44px', fontSize: '1rem' }}>
                                            {p.rating}
                                        </span>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* Event Timeline */}
            <div>
                <h3 style={{ fontSize: '1.3rem', fontWeight: 700, marginBottom: '16px' }}>Match Events</h3>
                <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
                    {events.slice(0, 50).map((e: any, i: number) => (
                        <div key={i} className="animate-slide-in" style={{
                            animationDelay: `${i * 0.02}s`, opacity: 0,
                            display: 'flex', gap: '12px', padding: '10px 16px',
                            background: i % 2 === 0 ? 'var(--bg-card)' : 'transparent',
                            borderRadius: '8px', alignItems: 'center',
                        }}>
                            <span style={{ color: 'var(--text-muted)', fontWeight: 600, fontSize: '0.85rem', minWidth: '40px' }}>
                                {formatEventTime(e)}
                            </span>
                            <span style={{
                                padding: '4px 10px', borderRadius: '6px', fontSize: '0.8rem', fontWeight: 600,
                                background: e.type === 'Shot' ? 'rgba(239,68,68,0.15)' : e.type === 'Pass' ? 'rgba(99,102,241,0.15)' : 'rgba(148,163,184,0.1)',
                                color: e.type === 'Shot' ? '#ef4444' : e.type === 'Pass' ? '#818cf8' : 'var(--text-secondary)',
                            }}>
                                {e.type}
                            </span>
                            <span style={{ fontWeight: 500 }}>{e.player || 'Unknown'}</span>
                            <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginLeft: 'auto' }}>{e.team}</span>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
