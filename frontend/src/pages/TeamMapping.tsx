import { useEffect, useMemo, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { confirmTeamMapping, detectTeamColors } from '../api/client';

type TeamColor = {
    team_id: number;
    detected_label: string;
    team_name?: string;
    color_name: string;
    hex: string;
};

export default function TeamMapping() {
    const { id } = useParams();
    const matchId = Number(id);
    const navigate = useNavigate();
    const [teamColors, setTeamColors] = useState<TeamColor[]>([]);
    const [teamNames, setTeamNames] = useState<Record<number, string>>({});
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [error, setError] = useState('');

    useEffect(() => {
        let mounted = true;
        setLoading(true);
        setError('');
        detectTeamColors(matchId)
            .then((res) => {
                if (!mounted) {
                    return;
                }
                const colors = Array.isArray(res.data.team_colors) ? res.data.team_colors : [];
                setTeamColors(colors);
                setTeamNames(Object.fromEntries(colors.map((team: TeamColor) => [team.team_id, ''])));
            })
            .catch((err) => {
                if (mounted) {
                    setError(err.response?.data?.detail || 'Could not detect team colors from this video.');
                }
            })
            .finally(() => {
                if (mounted) {
                    setLoading(false);
                }
            });
        return () => {
            mounted = false;
        };
    }, [matchId]);

    const canSave = useMemo(
        () => teamColors.length >= 2 && teamColors.every((team) => (teamNames[team.team_id] || '').trim()),
        [teamColors, teamNames],
    );

    const handleSwap = () => {
        if (teamColors.length < 2) {
            return;
        }
        const [first, second] = teamColors;
        setTeamNames((prev) => ({
            ...prev,
            [first.team_id]: prev[second.team_id] || '',
            [second.team_id]: prev[first.team_id] || '',
        }));
    };

    const handleConfirm = async () => {
        if (!canSave) {
            setError('Enter a team name for each detected color.');
            return;
        }
        setSaving(true);
        setError('');
        try {
            await confirmTeamMapping(matchId, teamNames);
            navigate(`/lineup/${matchId}`);
        } catch (err: any) {
            setError(err.response?.data?.detail || 'Failed to save team mapping.');
        } finally {
            setSaving(false);
        }
    };

    return (
        <div className="page-container" style={{ maxWidth: '900px', margin: '0 auto', paddingTop: '48px' }}>
            <h1 className="page-title">Confirm Team Colors</h1>
            <p className="page-subtitle">
                Component 1 detected the kit color groups. Assign the real team name to each color before lineup setup.
            </p>

            {error && (
                <div
                    style={{
                        background: 'rgba(239,68,68,0.1)',
                        border: '1px solid rgba(239,68,68,0.3)',
                        borderRadius: '12px',
                        padding: '12px 16px',
                        marginBottom: '24px',
                        color: '#ef4444',
                    }}
                >
                    {error}
                </div>
            )}

            {loading ? (
                <div className="glass-card" style={{ textAlign: 'center', padding: '48px' }}>
                    <div className="spinner" style={{ margin: '0 auto 16px' }} />
                    <div style={{ fontWeight: 700, marginBottom: '8px' }}>Detecting team colors...</div>
                    <p style={{ color: 'var(--text-secondary)', margin: 0 }}>
                        Sampling early match frames and clustering player kit colors.
                    </p>
                </div>
            ) : (
                <>
                    <div style={{ display: 'grid', gap: '16px', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))' }}>
                        {teamColors.map((team) => (
                            <div
                                key={team.team_id}
                                className="glass-card"
                                style={{
                                    borderRadius: '8px',
                                    border: '1px solid var(--border-subtle)',
                                }}
                            >
                                <div style={{ display: 'flex', alignItems: 'center', gap: '14px', marginBottom: '18px' }}>
                                    <span
                                        aria-hidden="true"
                                        style={{
                                            width: '42px',
                                            height: '42px',
                                            borderRadius: '50%',
                                            background: team.hex,
                                            border: '2px solid rgba(255,255,255,0.7)',
                                            flexShrink: 0,
                                        }}
                                    />
                                    <div>
                                        <div style={{ fontWeight: 800, color: 'var(--text-primary)' }}>{team.detected_label}</div>
                                        <div style={{ color: 'var(--text-secondary)', fontSize: '0.95rem' }}>
                                            {team.color_name} kit
                                        </div>
                                    </div>
                                </div>
                                <label style={{ display: 'grid', gap: '8px' }}>
                                    <span style={{ fontWeight: 600 }}>Real team name</span>
                                    <input
                                        type="text"
                                        value={teamNames[team.team_id] || ''}
                                        onChange={(e) => setTeamNames((prev) => ({ ...prev, [team.team_id]: e.target.value }))}
                                        placeholder={team.team_id === 0 ? 'e.g. Colombo Lions' : 'e.g. Kandy Blues'}
                                        disabled={saving}
                                        style={{
                                            borderRadius: '12px',
                                            padding: '12px',
                                            background: 'var(--bg-card)',
                                            color: 'var(--text-primary)',
                                            border: '1px solid var(--border-subtle)',
                                        }}
                                    />
                                </label>
                            </div>
                        ))}
                    </div>

                    <div style={{ display: 'flex', gap: '12px', marginTop: '24px', flexWrap: 'wrap' }}>
                        <button
                            type="button"
                            className="btn-secondary"
                            onClick={handleSwap}
                            disabled={saving || teamColors.length < 2}
                            style={{ flex: '1 1 180px', padding: '14px 24px' }}
                        >
                            Swap Team Names
                        </button>
                        <button
                            type="button"
                            className="btn-primary"
                            onClick={handleConfirm}
                            disabled={saving || !canSave}
                            style={{ flex: '2 1 260px', padding: '14px 24px' }}
                        >
                            {saving ? 'Saving...' : 'Confirm and Continue to Lineups ->'}
                        </button>
                    </div>
                </>
            )}
        </div>
    );
}
