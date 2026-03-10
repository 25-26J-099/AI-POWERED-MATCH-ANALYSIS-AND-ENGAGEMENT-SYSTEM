import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { getMatchStatus } from '../api/client';

const PIPELINE_STAGES = [
    { key: 'uploading', label: 'Uploading Video', icon: '📤' },
    { key: 'tracking', label: 'Tracking Players and Events', icon: '🎯' },
    { key: 'analytics_processing', label: 'Computing Analytics', icon: '📊' },
    { key: 'commentary_generation', label: 'Preparing Commentary Export', icon: '🎙️' },
    { key: 'completed', label: 'Analysis Complete', icon: '✅' },
];

export default function Processing() {
    const { id } = useParams();
    const navigate = useNavigate();
    const [status, setStatus] = useState('uploading');
    const [detail, setDetail] = useState('');
    const [error, setError] = useState(false);

    useEffect(() => {
        const poll = setInterval(async () => {
            try {
                const res = await getMatchStatus(Number(id));
                setStatus(res.data.status);
                setDetail(res.data.status_detail || '');
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

    const currentIndex = PIPELINE_STAGES.findIndex((stage) => stage.key === status);

    return (
        <div className="page-container" style={{ maxWidth: '700px', margin: '0 auto', paddingTop: '48px' }}>
            <h1 className="page-title" style={{ textAlign: 'center' }}>Processing Match</h1>
            <p className="page-subtitle" style={{ textAlign: 'center' }}>
                Component 1 tracking and Component 4 analytics are running in sequence.
            </p>

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
                                {state === 'active' && detail && (
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
