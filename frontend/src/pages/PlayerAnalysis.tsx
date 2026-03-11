import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { getPlayerDetail } from '../api/client';
import RadarChart from '../charts/RadarChart';
import HeatmapCanvas from '../charts/HeatmapCanvas';

export default function PlayerAnalysis() {
    const { id, playerId } = useParams();
    const [data, setData] = useState<any>(null);

    useEffect(() => {
        getPlayerDetail(Number(id), Number(playerId)).then(r => setData(r.data));
    }, [id, playerId]);

    if (!data) return <div className="page-container"><div className="spinner" style={{ margin: '80px auto' }} /></div>;

    const stats = data.stats;
    const ratingClass = data.rating >= 8 ? 'rating-excellent' : data.rating >= 6 ? 'rating-good' : data.rating >= 4 ? 'rating-average' : 'rating-poor';

    return (
        <div className="page-container">
            {/* Player Header */}
            <div className="glass-card" style={{ display: 'flex', alignItems: 'center', gap: '24px', marginBottom: '32px' }}>
                <div className={`rating-badge ${ratingClass}`}>{data.rating}</div>
                <div>
                    <h1 style={{ fontSize: '1.8rem', fontWeight: 800 }}>{data.name}</h1>
                    <p style={{ color: 'var(--text-secondary)' }}>{data.team} • {data.position || 'N/A'}</p>
                </div>
                {data.style_cluster !== null && (
                    <div style={{
                        marginLeft: 'auto', padding: '8px 16px', borderRadius: '20px',
                        background: 'rgba(99,102,241,0.15)', color: 'var(--accent-light)',
                        fontSize: '0.85rem', fontWeight: 600,
                    }}>
                        {data.style_cluster_label || `Cluster ${data.style_cluster}`}
                    </div>
                )}
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', marginBottom: '32px' }}>
                {/* Radar Chart */}
                <div className="glass-card">
                    <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '16px' }}>Player Profile</h3>
                    <RadarChart data={[{
                        name: data.name,
                        values: {
                            Passing: Math.min(stats.pass_accuracy, 100),
                            Shooting: Math.min(stats.xg * 100, 100),
                            Threat: Math.min(stats.xt * 200 + 50, 100),
                            VAEP: Math.min((stats.vaep + 0.5) * 50, 100),
                            Progressive: Math.min((stats.progressive_passes / Math.max(stats.passes, 1)) * 300, 100),
                            Defensive: Math.min(stats.pressures * 5, 100),
                        },
                    }]} />
                </div>

                {/* Heatmap */}
                <div className="glass-card">
                    <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '16px' }}>Position Heatmap</h3>
                    <HeatmapCanvas data={data.heatmap} />
                </div>
            </div>

            {/* Stats Grid */}
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '16px' }}>Detailed Stats</h3>
            <div className="stats-grid">
                {[
                    { label: 'Passes', value: stats.passes },
                    { label: 'Pass Accuracy', value: `${stats.pass_accuracy}%` },
                    { label: 'Progressive Passes', value: stats.progressive_passes },
                    { label: 'Carries', value: stats.carries },
                    { label: 'Shots', value: stats.shots },
                    { label: 'Touches', value: stats.touches },
                    { label: 'Pressures', value: stats.pressures },
                    { label: 'Recoveries', value: stats.recoveries },
                    { label: 'xG', value: stats.xg.toFixed(3) },
                    { label: 'xT', value: stats.xt.toFixed(3) },
                    { label: 'VAEP', value: stats.vaep.toFixed(3) },
                    { label: 'Rating', value: data.rating },
                ].map((s, i) => (
                    <div key={i} className="stat-card animate-fade-in-up" style={{ animationDelay: `${i * 0.05}s`, opacity: 0 }}>
                        <div className="stat-value">{s.value}</div>
                        <div className="stat-label">{s.label}</div>
                    </div>
                ))}
            </div>
        </div>
    );
}
