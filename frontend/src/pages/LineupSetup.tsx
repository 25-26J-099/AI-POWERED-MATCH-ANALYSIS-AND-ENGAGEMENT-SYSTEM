import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import LineupBuilder from '../components/LineupBuilder';
import { getMatch, submitLineups, proceedPipeline } from '../api/client';

export default function LineupSetup() {
    const { id } = useParams();
    const matchId = Number(id);
    const navigate = useNavigate();
    const [submitting, setSubmitting] = useState(false);
    const [error, setError] = useState('');
    const [teamNames, setTeamNames] = useState({ home: '', away: '' });

    useEffect(() => {
        if (!matchId) {
            return;
        }
        getMatch(matchId)
            .then((res) => setTeamNames({
                home: res.data.home_team || '',
                away: res.data.away_team || '',
            }))
            .catch(() => { });
    }, [matchId]);

    const handleSubmit = async (data: any) => {
        setSubmitting(true);
        setError('');
        try {
            await submitLineups(matchId, data);
            await proceedPipeline(matchId);
            navigate(`/processing/${matchId}`);
        } catch (err: any) {
            setError(err.response?.data?.detail || 'Failed to submit lineups');
        } finally {
            setSubmitting(false);
        }
    };

    const handleSkip = async () => {
        setSubmitting(true);
        setError('');
        try {
            await proceedPipeline(matchId);
            navigate(`/processing/${matchId}`);
        } catch (err: any) {
            setError(err.response?.data?.detail || 'Failed to start pipeline');
        } finally {
            setSubmitting(false);
        }
    };

    return (
        <div className="page-container" style={{ maxWidth: '1000px', margin: '0 auto', paddingTop: '48px' }}>
            <h1 className="page-title">Lineup Setup</h1>
            <p className="page-subtitle">
                Enter team lineups and formations before processing. You can also skip this step.
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

            <div className="glass-card" style={{ marginBottom: '24px' }}>
                <LineupBuilder
                    onSubmit={handleSubmit}
                    initialHomeTeamName={teamNames.home}
                    initialAwayTeamName={teamNames.away}
                />
            </div>

            {submitting && (
                <div style={{ textAlign: 'center', marginBottom: '16px' }}>
                    <div className="spinner" style={{ margin: '0 auto 8px' }} />
                    <p style={{ color: 'var(--text-secondary)' }}>Starting pipeline...</p>
                </div>
            )}

            <div style={{ textAlign: 'center' }}>
                <button
                    className="btn-secondary"
                    onClick={handleSkip}
                    disabled={submitting}
                    style={{ padding: '14px 32px', fontSize: '1rem' }}
                >
                    Skip Lineup Setup →
                </button>
            </div>
        </div>
    );
}
