import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { getMatchStatus } from '../api/client';

const PIPELINE_STAGES = [
    { key: 'lineup_pending', label: 'Lineup Setup', icon: '📋' },
    { key: 'uploading', label: 'Uploading Video', icon: '📤' },
    { key: 'tracking', label: 'Tracking Players and Events', icon: '🎯' },
    { key: 'analytics_processing', label: 'Computing Analytics', icon: '📊' },
    { key: 'play_by_play_commentary', label: 'Generating Play-by-Play Commentary', icon: '🎙️' },
    { key: 'expert_commentary', label: 'Processing Tactical Insights', icon: '🧠' },
    { key: 'video_rendering', label: 'Rendering Commentary Video', icon: '🎬' },
    { key: 'completed', label: 'Analysis Complete', icon: '✅' },
];

const PIPELINE_STAGE_KEYS = new Set(PIPELINE_STAGES.map((stage) => stage.key));

export default function Processing() {
    const { id } = useParams();
    const navigate = useNavigate();
    const [status, setStatus] = useState('uploading');
    const [lastActiveStage, setLastActiveStage] = useState('uploading');
    const [detail, setDetail] = useState('');
    const [teamColors, setTeamColors] = useState<any[]>([]);
    const [error, setError] = useState(false);

    useEffect(() => {
        const poll = setInterval(async () => {
            try {
                const res = await getMatchStatus(Number(id));
                setStatus(res.data.status);
                setDetail(res.data.status_detail || '');
                if (Array.isArray(res.data.team_colors)) {
                    setTeamColors(res.data.team_colors);
                }
                if (PIPELINE_STAGE_KEYS.has(res.data.status) && res.data.status !== 'completed') {
                    setLastActiveStage(res.data.status);
                }
                if (res.data.status === 'completed') {
                    clearInterval(poll);
                    setTimeout(() => navigate(`/match/${id}`), 1500);
                }
                if (res.data.status === 'failed') {
                    clearInterval(poll);
                    setError(true);
                }
            } catch {
                /* ignore polling errors */
            }
        }, 2000);
        return () => clearInterval(poll);
    }, [id, navigate]);

    const displayedStatus = status === 'failed' ? lastActiveStage : status;
    const currentIndex = PIPELINE_STAGES.findIndex((stage) => stage.key === displayedStatus);

    return (
        <div className="page-container" style={{ maxWidth: '700px', margin: '0 auto', paddingTop: '48px' }}>
            <h1 className="page-title" style={{ textAlign: 'center' }}>Processing Match</h1>
            

            {error && (
                <div
                    className="card"
                    style={{
                        marginTop: '24px',
                        borderColor: 'var(--danger, #c53030)',
                        background: 'rgba(197, 48, 48, 0.08)',
                    }}
                >
                    <div style={{ fontWeight: 700, color: 'var(--text-primary)' }}>Pipeline failed</div>
                    <div style={{ marginTop: '8px', color: 'var(--text-secondary)' }}>
                        {detail || 'The backend reported a failure while processing this match.'}
                    </div>
                </div>
            )}

            {teamColors.length > 0 && (
                <div className="glass-card" style={{ marginTop: '28px' }}>
                    <h3 style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: '14px' }}>Detected Team Colors</h3>
                    <div style={{ display: 'grid', gap: '12px', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))' }}>
                        {teamColors.map((team) => (
                            <div
                                key={team.team_id}
                                style={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '12px',
                                    padding: '12px',
                                    borderRadius: '8px',
                                    background: 'var(--bg-card)',
                                    border: '1px solid var(--border-subtle)',
                                }}
                            >
                                <span
                                    aria-hidden="true"
                                    style={{
                                        width: '28px',
                                        height: '28px',
                                        borderRadius: '50%',
                                        background: team.hex,
                                        border: '2px solid rgba(255,255,255,0.65)',
                                        flexShrink: 0,
                                    }}
                                />
                                <div>
                                    <div style={{ fontWeight: 700, color: 'var(--text-primary)' }}>
                                        {team.team_name || team.detected_label}
                                    </div>
                                    <div style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                                        {team.detected_label} - {team.color_name}
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginTop: '40px' }}>
                {PIPELINE_STAGES.map((stage, index) => {
                    let state: 'pending' | 'active' | 'completed' | 'failed' = 'pending';
                    if (error && index === currentIndex) {
                        state = 'failed';
                    } else if (index < currentIndex) {
                        state = 'completed';
                    } else if (index === currentIndex) {
                        state = status === 'completed' ? 'completed' : 'active';
                    }

                    return (
                        <div key={stage.key} className={`pipeline-step ${state}`} style={{ animationDelay: `${index * 0.1}s` }}>
                            <div className={`step-indicator ${state}`}>
                                {state === 'completed' ? '✓' : state === 'failed' ? '✗' : stage.icon}
                            </div>
                            <div>
                                <div
                                    style={{
                                        fontWeight: 600,
                                        color: state === 'pending' ? 'var(--text-muted)' : 'var(--text-primary)',
                                    }}
                                >
                                    {stage.label}
                                </div>
                                {(state === 'active' || state === 'failed') && detail && index === currentIndex && (
                                    <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                                        {detail}
                                    </div>
                                )}
                            </div>
                            {state === 'active' && <div className="spinner" style={{ marginLeft: 'auto' }} />}
                        </div>
                    );
                })}
            </div>

            {status === 'completed' && (
                <div style={{ textAlign: 'center', marginTop: '32px' }} className="animate-fade-in-up">
                    <button className="btn-primary" onClick={() => navigate(`/match/${id}`)}>
                        View Match Analysis →
                    </button>
                </div>
            )}
        </div>
    );
}
