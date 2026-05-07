import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { getAiAnalysis } from '../api/client';

export default function AiAnalysis() {
    const { id } = useParams();
    const [analysis, setAnalysis] = useState<any>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const generate = async () => {
        setLoading(true);
        setError('');
        try {
            const res = await getAiAnalysis(Number(id));
            setAnalysis(res.data.analysis);
        } catch (err: any) {
            setError(err.response?.data?.detail || 'Failed to generate analysis');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => { generate(); }, [id]);

    const sections = analysis ? [
        { title: '📋 Match Overview', content: analysis.match_overview },
        { title: '🧠 Tactical Analysis', content: analysis.tactical_analysis },
        { title: '⚡ Key Moments', content: analysis.key_moments },
        { title: '🏆 Player of the Match', content: analysis.player_of_the_match },
        { title: '⭐ Top Performers', content: analysis.top_performers },
        { title: '⚔️ Team Comparison', content: analysis.team_comparison },
        { title: '📈 xG Analysis', content: analysis.xg_analysis },
        { title: '💎 Possession Quality', content: analysis.possession_quality },
        { title: '🔧 Areas to Improve', content: analysis.areas_to_improve },
    ].filter(s => s.content) : [];

    return (
        <div className="page-container" style={{ maxWidth: '900px', margin: '0 auto' }}>
            <h1 className="page-title">AI Expert Analysis</h1>
            <p className="page-subtitle">GPT-powered expert analysis using match analytics data</p>

            {loading && (
                <div className="glass-card" style={{ textAlign: 'center', padding: '80px' }}>
                    <div className="spinner" style={{ margin: '0 auto 16px' }} />
                    <p style={{ color: 'var(--text-secondary)' }}>Generating expert analysis...</p>
                </div>
            )}

            {error && (
                <div style={{
                    background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)',
                    borderRadius: '12px', padding: '16px', marginBottom: '24px', color: '#ef4444',
                }}>
                    {error}
                    <button className="btn-secondary" onClick={generate} style={{ marginLeft: '16px' }}>Retry</button>
                </div>
            )}

            {/* Fallback analysis */}
            {analysis?.fallback_analysis && (
                <div className="glass-card" style={{ marginBottom: '24px' }}>
                    <div style={{
                        padding: '8px 14px', borderRadius: '8px', background: 'rgba(245,158,11,0.1)',
                        color: 'var(--warning)', fontSize: '0.85rem', marginBottom: '16px',
                    }}>
                        ⚠️ {analysis.fallback_analysis.note}
                    </div>
                    <p>{analysis.fallback_analysis.match_overview}</p>
                    <p style={{ marginTop: '8px', color: 'var(--text-secondary)' }}>
                        POTM: <strong>{analysis.fallback_analysis.player_of_the_match}</strong>
                    </p>
                </div>
            )}

            {/* Full analysis sections */}
            {sections.map((s, i) => (
                <div key={i} className="glass-card animate-fade-in-up" style={{ marginBottom: '16px', animationDelay: `${i * 0.1}s`, opacity: 0 }}>
                    <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '12px' }}>{s.title}</h3>
                    <p style={{ color: 'var(--text-secondary)', lineHeight: 1.8, whiteSpace: 'pre-wrap' }}>
                        {typeof s.content === 'string' ? s.content : JSON.stringify(s.content, null, 2)}
                    </p>
                </div>
            ))}
        </div>
    );
}
