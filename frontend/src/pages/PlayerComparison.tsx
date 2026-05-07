import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { getMatchPlayers, comparePlayersApi } from '../api/client';
import RadarChart from '../charts/RadarChart';

export default function PlayerComparison() {
    const { id } = useParams();
    const matchId = Number(id);
    const [players, setPlayers] = useState<any[]>([]);
    const [p1, setP1] = useState<number | null>(null);
    const [p2, setP2] = useState<number | null>(null);
    const [comparison, setComparison] = useState<any>(null);

    useEffect(() => {
        getMatchPlayers(matchId).then(r => setPlayers(r.data));
    }, [matchId]);

    useEffect(() => {
        if (p1 && p2) {
            comparePlayersApi(matchId, p1, p2).then(r => setComparison(r.data));
        }
    }, [matchId, p1, p2]);

    const selectStyle = {
        padding: '12px 16px', borderRadius: '10px', background: 'var(--bg-secondary)',
        border: '1px solid var(--border-subtle)', color: 'var(--text-primary)',
        fontSize: '1rem', outline: 'none', width: '100%', cursor: 'pointer',
    };

    const toRadarValues = (s: any) => ({
        Passing: Math.min(s.pass_accuracy, 100),
        Shooting: Math.min(s.xg * 100, 100),
        Threat: Math.min(s.xt * 200 + 50, 100),
        VAEP: Math.min((s.vaep + 0.5) * 50, 100),
        Progressive: Math.min((s.progressive_passes / Math.max(s.passes, 1)) * 300, 100),
        Defensive: Math.min(s.pressures * 5, 100),
    });

    return (
        <div className="page-container" style={{ maxWidth: '1000px', margin: '0 auto' }}>
            <h1 className="page-title">Player Comparison</h1>
            <p className="page-subtitle">Select two players to compare side by side</p>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', marginBottom: '32px' }}>
                <div>
                    <label style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '6px', display: 'block' }}>Player 1</label>
                    <select value={p1 || ''} onChange={e => setP1(Number(e.target.value))} style={selectStyle as any}>
                        <option value="">Select player...</option>
                        {players.map((p: any) => <option key={p.player_id} value={p.player_id}>{p.name} ({p.team})</option>)}
                    </select>
                </div>
                <div>
                    <label style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '6px', display: 'block' }}>Player 2</label>
                    <select value={p2 || ''} onChange={e => setP2(Number(e.target.value))} style={selectStyle as any}>
                        <option value="">Select player...</option>
                        {players.map((p: any) => <option key={p.player_id} value={p.player_id}>{p.name} ({p.team})</option>)}
                    </select>
                </div>
            </div>

            {comparison && (
                <div className="animate-fade-in-up">
                    {/* Overlay Radar */}
                    <div className="glass-card" style={{ marginBottom: '24px' }}>
                        <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '16px' }}>Profile Overlay</h3>
                        <RadarChart data={[
                            { name: comparison.player1.name, values: toRadarValues(comparison.player1) },
                            { name: comparison.player2.name, values: toRadarValues(comparison.player2) },
                        ]} />
                    </div>

                    {/* Stat Diff Table */}
                    <div className="glass-card" style={{ padding: '0', overflow: 'hidden' }}>
                        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                            <thead>
                                <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                                    <th style={{ padding: '14px 16px', textAlign: 'right', color: 'var(--accent-light)', fontWeight: 700 }}>{comparison.player1.name}</th>
                                    <th style={{ padding: '14px 16px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.85rem' }}>Stat</th>
                                    <th style={{ padding: '14px 16px', textAlign: 'left', color: 'var(--accent-light)', fontWeight: 700 }}>{comparison.player2.name}</th>
                                </tr>
                            </thead>
                            <tbody>
                                {[
                                    { label: 'Rating', k: 'rating' },
                                    { label: 'Passes', k: 'passes' },
                                    { label: 'Pass Accuracy', k: 'pass_accuracy', suffix: '%' },
                                    { label: 'Shots', k: 'shots' },
                                    { label: 'xG', k: 'xg', dec: 3 },
                                    { label: 'xT', k: 'xt', dec: 3 },
                                    { label: 'VAEP', k: 'vaep', dec: 3 },
                                    { label: 'Pressures', k: 'pressures' },
                                    { label: 'Touches', k: 'touches' },
                                ].map((row, i) => {
                                    const v1 = comparison.player1[row.k];
                                    const v2 = comparison.player2[row.k];
                                    const fmt = (v: number) => (row.dec ? v.toFixed(row.dec) : v) + (row.suffix || '');
                                    const better1 = v1 > v2;
                                    const better2 = v2 > v1;
                                    return (
                                        <tr key={i} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                                            <td style={{ padding: '12px 16px', textAlign: 'right', fontWeight: better1 ? 700 : 400, color: better1 ? 'var(--success)' : 'var(--text-primary)' }}>{fmt(v1)}</td>
                                            <td style={{ padding: '12px 16px', textAlign: 'center', color: 'var(--text-secondary)', fontSize: '0.85rem' }}>{row.label}</td>
                                            <td style={{ padding: '12px 16px', fontWeight: better2 ? 700 : 400, color: better2 ? 'var(--success)' : 'var(--text-primary)' }}>{fmt(v2)}</td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}
        </div>
    );
}
