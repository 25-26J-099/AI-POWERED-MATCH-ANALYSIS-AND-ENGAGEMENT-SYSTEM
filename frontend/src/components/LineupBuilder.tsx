import { useEffect, useState } from 'react';

interface Props {
    onSubmit: (data: any) => void;
    initialHomeTeamName?: string;
    initialAwayTeamName?: string;
}

const FORMATIONS: Record<string, { label: string; positions: { x: number; y: number; role: string }[] }> = {
    '4-3-3': {
        label: '4-3-3',
        positions: [
            { x: 50, y: 90, role: 'GK' },
            { x: 20, y: 72, role: 'LB' }, { x: 40, y: 75, role: 'CB' }, { x: 60, y: 75, role: 'CB' }, { x: 80, y: 72, role: 'RB' },
            { x: 30, y: 50, role: 'CM' }, { x: 50, y: 45, role: 'CM' }, { x: 70, y: 50, role: 'CM' },
            { x: 20, y: 22, role: 'LW' }, { x: 50, y: 18, role: 'ST' }, { x: 80, y: 22, role: 'RW' },
        ],
    },
    '4-4-2': {
        label: '4-4-2',
        positions: [
            { x: 50, y: 90, role: 'GK' },
            { x: 20, y: 72, role: 'LB' }, { x: 40, y: 75, role: 'CB' }, { x: 60, y: 75, role: 'CB' }, { x: 80, y: 72, role: 'RB' },
            { x: 15, y: 48, role: 'LM' }, { x: 38, y: 50, role: 'CM' }, { x: 62, y: 50, role: 'CM' }, { x: 85, y: 48, role: 'RM' },
            { x: 38, y: 22, role: 'ST' }, { x: 62, y: 22, role: 'ST' },
        ],
    },
    '3-5-2': {
        label: '3-5-2',
        positions: [
            { x: 50, y: 90, role: 'GK' },
            { x: 30, y: 75, role: 'CB' }, { x: 50, y: 77, role: 'CB' }, { x: 70, y: 75, role: 'CB' },
            { x: 12, y: 50, role: 'LWB' }, { x: 35, y: 52, role: 'CM' }, { x: 50, y: 48, role: 'CM' }, { x: 65, y: 52, role: 'CM' }, { x: 88, y: 50, role: 'RWB' },
            { x: 38, y: 22, role: 'ST' }, { x: 62, y: 22, role: 'ST' },
        ],
    },
    '4-2-3-1': {
        label: '4-2-3-1',
        positions: [
            { x: 50, y: 90, role: 'GK' },
            { x: 20, y: 72, role: 'LB' }, { x: 40, y: 75, role: 'CB' }, { x: 60, y: 75, role: 'CB' }, { x: 80, y: 72, role: 'RB' },
            { x: 38, y: 55, role: 'CDM' }, { x: 62, y: 55, role: 'CDM' },
            { x: 20, y: 35, role: 'LW' }, { x: 50, y: 33, role: 'CAM' }, { x: 80, y: 35, role: 'RW' },
            { x: 50, y: 15, role: 'ST' },
        ],
    },
};

type TeamData = {
    team_name: string;
    formation: string;
    players: { player_name: string; jersey_number: string; position_slot: number }[];
};

export default function LineupBuilder({ onSubmit, initialHomeTeamName = '', initialAwayTeamName = '' }: Props) {
    const [homeTeam, setHomeTeam] = useState<TeamData>({
        team_name: '',
        formation: '4-3-3',
        players: Array.from({ length: 11 }, (_, i) => ({ player_name: '', jersey_number: '', position_slot: i })),
    });
    const [awayTeam, setAwayTeam] = useState<TeamData>({
        team_name: '',
        formation: '4-3-3',
        players: Array.from({ length: 11 }, (_, i) => ({ player_name: '', jersey_number: '', position_slot: i })),
    });
    const [activeTeam, setActiveTeam] = useState<'home' | 'away'>('home');

    useEffect(() => {
        if (initialHomeTeamName) {
            setHomeTeam(prev => prev.team_name ? prev : { ...prev, team_name: initialHomeTeamName });
        }
        if (initialAwayTeamName) {
            setAwayTeam(prev => prev.team_name ? prev : { ...prev, team_name: initialAwayTeamName });
        }
    }, [initialHomeTeamName, initialAwayTeamName]);

    const current = activeTeam === 'home' ? homeTeam : awayTeam;
    const setCurrent = activeTeam === 'home' ? setHomeTeam : setAwayTeam;
    const formation = FORMATIONS[current.formation];

    const updatePlayer = (index: number, field: 'player_name' | 'jersey_number', value: string) => {
        setCurrent(prev => ({
            ...prev,
            players: prev.players.map((p, i) => i === index ? { ...p, [field]: value } : p),
        }));
    };

    const handleSubmit = () => {
        const data = {
            home_team: {
                team_name: homeTeam.team_name.trim() || 'Team 1',
                formation: homeTeam.formation,
                players: homeTeam.players.map(p => ({
                    player_name: p.player_name || `Player ${p.position_slot + 1}`,
                    jersey_number: parseInt(p.jersey_number) || p.position_slot + 1,
                    position_slot: p.position_slot,
                })),
            },
            away_team: {
                team_name: awayTeam.team_name.trim() || 'Team 2',
                formation: awayTeam.formation,
                players: awayTeam.players.map(p => ({
                    player_name: p.player_name || `Player ${p.position_slot + 1}`,
                    jersey_number: parseInt(p.jersey_number) || p.position_slot + 1,
                    position_slot: p.position_slot,
                })),
            },
        };
        onSubmit(data);
    };

    return (
        <div>
            <h3 style={{ fontSize: '1.3rem', fontWeight: 700, marginBottom: '16px' }}>Set Up Lineups</h3>

            {/* Team Tabs */}
            <div style={{ display: 'flex', gap: '8px', marginBottom: '24px' }}>
                {(['home', 'away'] as const).map(t => (
                    <button key={t} onClick={() => setActiveTeam(t)}
                        style={{
                            flex: 1, padding: '12px', borderRadius: '12px', fontWeight: 600,
                            background: activeTeam === t ? 'var(--accent)' : 'var(--bg-card)',
                            color: activeTeam === t ? 'white' : 'var(--text-secondary)',
                            border: '1px solid ' + (activeTeam === t ? 'var(--accent)' : 'var(--border-subtle)'),
                            cursor: 'pointer', transition: 'all 0.2s', textTransform: 'capitalize',
                        }}>
                        {t} Team
                    </button>
                ))}
            </div>

            {/* Team Name */}
            <div style={{ marginBottom: '20px' }}>
                <label style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', display: 'block', marginBottom: '6px' }}>
                    Team Name
                </label>
                <input
                    type="text"
                    value={current.team_name}
                    onChange={e => setCurrent(prev => ({ ...prev, team_name: e.target.value }))}
                    placeholder={`${activeTeam === 'home' ? 'Home' : 'Away'} team name`}
                    style={{
                        width: '100%', padding: '12px 16px', borderRadius: '10px',
                        background: 'var(--bg-secondary)', border: '1px solid var(--border-subtle)',
                        color: 'var(--text-primary)', fontSize: '1rem', outline: 'none',
                    }}
                />
            </div>

            {/* Formation Selector */}
            <div style={{ marginBottom: '24px' }}>
                <label style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', display: 'block', marginBottom: '6px' }}>
                    Formation
                </label>
                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                    {Object.keys(FORMATIONS).map(f => (
                        <button key={f} onClick={() => setCurrent(prev => ({ ...prev, formation: f }))}
                            style={{
                                padding: '8px 20px', borderRadius: '8px', fontWeight: 600,
                                background: current.formation === f ? 'var(--accent)' : 'var(--bg-card)',
                                color: current.formation === f ? 'white' : 'var(--text-secondary)',
                                border: '1px solid ' + (current.formation === f ? 'var(--accent)' : 'var(--border-subtle)'),
                                cursor: 'pointer', transition: 'all 0.2s',
                            }}>
                            {f}
                        </button>
                    ))}
                </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
                {/* Formation Pitch Preview */}
                <div className="formation-pitch">
                    {/* Pitch markings */}
                    <div style={{ position: 'absolute', top: '50%', left: '10%', right: '10%', height: '1px', background: 'rgba(255,255,255,0.2)' }} />
                    <div style={{ position: 'absolute', top: '50%', left: '50%', width: '60px', height: '60px', borderRadius: '50%', border: '1px solid rgba(255,255,255,0.2)', transform: 'translate(-50%,-50%)' }} />

                    {formation.positions.map((pos, i) => (
                        <div key={i} className="formation-player-dot"
                            style={{ left: `${pos.x}%`, top: `${pos.y}%` }}
                            title={`${pos.role} — ${current.players[i]?.player_name || 'Empty'}`}>
                            {current.players[i]?.jersey_number || (i + 1)}
                        </div>
                    ))}
                </div>

                {/* Player Inputs */}
                <div style={{ maxHeight: '500px', overflowY: 'auto', paddingRight: '8px' }}>
                    {formation.positions.map((pos, i) => (
                        <div key={i} style={{
                            display: 'grid', gridTemplateColumns: '50px 1fr 70px', gap: '8px',
                            marginBottom: '8px', alignItems: 'center',
                        }}>
                            <span style={{
                                color: 'var(--accent-light)', fontWeight: 700, fontSize: '0.8rem',
                                background: 'rgba(99,102,241,0.1)', padding: '6px', borderRadius: '6px', textAlign: 'center',
                            }}>
                                {pos.role}
                            </span>
                            <input
                                type="text"
                                placeholder="Player name"
                                value={current.players[i]?.player_name || ''}
                                onChange={e => updatePlayer(i, 'player_name', e.target.value)}
                                style={{
                                    padding: '8px 12px', borderRadius: '8px', background: 'var(--bg-secondary)',
                                    border: '1px solid var(--border-subtle)', color: 'var(--text-primary)',
                                    fontSize: '0.9rem', outline: 'none',
                                }}
                            />
                            <input
                                type="number"
                                placeholder="#"
                                value={current.players[i]?.jersey_number || ''}
                                onChange={e => updatePlayer(i, 'jersey_number', e.target.value)}
                                style={{
                                    padding: '8px', borderRadius: '8px', background: 'var(--bg-secondary)',
                                    border: '1px solid var(--border-subtle)', color: 'var(--text-primary)',
                                    fontSize: '0.9rem', textAlign: 'center', outline: 'none',
                                }}
                            />
                        </div>
                    ))}
                </div>
            </div>

            <button className="btn-primary" onClick={handleSubmit}
                style={{ width: '100%', marginTop: '24px', padding: '16px' }}>
                Confirm Lineups →
            </button>
        </div>
    );
}
