import { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { uploadVideo, submitLineups, proceedPipeline } from '../api/client';
import LineupBuilder from '../components/LineupBuilder';

type Step = 'upload' | 'lineups' | 'ready';

export default function Upload() {
    const navigate = useNavigate();
    const [step, setStep] = useState<Step>('upload');
    const [file, setFile] = useState<File | null>(null);
    const [progress, setProgress] = useState(0);
    const [matchId, setMatchId] = useState<number | null>(null);
    const [uploading, setUploading] = useState(false);
    const [error, setError] = useState('');
    const [dragOver, setDragOver] = useState(false);

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setDragOver(false);
        const f = e.dataTransfer.files[0];
        if (f && f.type.startsWith('video/')) setFile(f);
        else setError('Please upload a video file.');
    }, []);

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        const f = e.target.files?.[0];
        if (f) { setFile(f); setError(''); }
    };

    const handleUpload = async () => {
        if (!file) return;
        setUploading(true);
        setError('');
        try {
            const res = await uploadVideo(file, setProgress);
            setMatchId(res.data.match_id);
            setStep('lineups');
        } catch (err: any) {
            setError(err.response?.data?.detail || 'Upload failed');
        } finally {
            setUploading(false);
        }
    };

    const handleLineupsSubmit = async (lineupData: any) => {
        if (!matchId) return;
        try {
            await submitLineups(matchId, lineupData);
            setStep('ready');
        } catch (err: any) {
            setError(err.response?.data?.detail || 'Failed to submit lineups');
        }
    };

    const handleProceed = async () => {
        if (!matchId) return;
        try {
            await proceedPipeline(matchId);
            navigate(`/processing/${matchId}`);
        } catch (err: any) {
            setError(err.response?.data?.detail || 'Failed to start pipeline');
        }
    };

    return (
        <div className="page-container" style={{ maxWidth: '900px', margin: '0 auto', paddingTop: '48px' }}>
            <h1 className="page-title">Upload Match Video</h1>
            <p className="page-subtitle">Upload your video, set up lineups, and let AI analyze the match</p>

            {/* Step Indicator */}
            <div style={{ display: 'flex', gap: '4px', marginBottom: '40px' }}>
                {['Upload Video', 'Set Lineups', 'Start Analysis'].map((label, i) => {
                    const stepMap: Step[] = ['upload', 'lineups', 'ready'];
                    const isActive = stepMap.indexOf(step) >= i;
                    return (
                        <div key={i} style={{ flex: 1 }}>
                            <div style={{
                                height: '4px', borderRadius: '2px',
                                background: isActive ? 'var(--accent)' : 'var(--bg-card)',
                                transition: 'background 0.3s',
                            }} />
                            <div style={{
                                fontSize: '0.8rem', marginTop: '8px',
                                color: isActive ? 'var(--accent-light)' : 'var(--text-muted)',
                                fontWeight: isActive ? 600 : 400,
                            }}>{label}</div>
                        </div>
                    );
                })}
            </div>

            {error && (
                <div style={{
                    background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)',
                    borderRadius: '12px', padding: '12px 16px', marginBottom: '24px', color: '#ef4444',
                }}>
                    {error}
                </div>
            )}

            {/* Step 1: Upload */}
            {step === 'upload' && (
                <div className="animate-fade-in-up">
                    <div
                        className="glass-card"
                        onDrop={handleDrop}
                        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                        onDragLeave={() => setDragOver(false)}
                        style={{
                            border: dragOver ? '2px dashed var(--accent)' : '2px dashed var(--border-subtle)',
                            textAlign: 'center', padding: '60px 40px', cursor: 'pointer',
                            background: dragOver ? 'rgba(99,102,241,0.05)' : 'var(--bg-glass)',
                            transition: 'all 0.3s',
                        }}
                        onClick={() => document.getElementById('video-input')?.click()}
                    >
                        <div style={{ fontSize: '4rem', marginBottom: '16px' }}>🎬</div>
                        <h3 style={{ fontSize: '1.3rem', fontWeight: 600, marginBottom: '8px' }}>
                            {file ? file.name : 'Drop your match video here'}
                        </h3>
                        <p style={{ color: 'var(--text-secondary)' }}>
                            {file ? `${(file.size / 1024 / 1024).toFixed(1)} MB` : 'or click to browse • MP4, AVI, MOV'}
                        </p>
                        <input id="video-input" type="file" accept="video/*" style={{ display: 'none' }} onChange={handleFileSelect} />
                    </div>

                    {file && (
                        <div style={{ marginTop: '24px' }}>
                            {uploading && (
                                <div style={{ marginBottom: '16px' }}>
                                    <div style={{
                                        height: '8px', borderRadius: '4px', background: 'var(--bg-card)', overflow: 'hidden',
                                    }}>
                                        <div style={{
                                            height: '100%', width: `${progress}%`, background: 'var(--gradient-primary)',
                                            borderRadius: '4px', transition: 'width 0.3s',
                                        }} />
                                    </div>
                                    <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: '8px' }}>
                                        Uploading... {progress}%
                                    </p>
                                </div>
                            )}
                            <button className="btn-primary" onClick={handleUpload} disabled={uploading}
                                style={{ width: '100%', padding: '16px' }}>
                                {uploading ? 'Uploading...' : 'Upload Video →'}
                            </button>
                        </div>
                    )}
                </div>
            )}

            {/* Step 2: Lineups */}
            {step === 'lineups' && (
                <div className="animate-fade-in-up">
                    <LineupBuilder onSubmit={handleLineupsSubmit} />
                </div>
            )}

            {/* Step 3: Ready to Proceed */}
            {step === 'ready' && (
                <div className="animate-fade-in-up" style={{ textAlign: 'center' }}>
                    <div className="glass-card" style={{ padding: '60px 40px' }}>
                        <div style={{ fontSize: '4rem', marginBottom: '16px' }}>✅</div>
                        <h3 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: '8px' }}>Ready to Analyze</h3>
                        <p style={{ color: 'var(--text-secondary)', marginBottom: '32px' }}>
                            Video uploaded and lineups configured. Click below to start the AI analysis pipeline.
                        </p>
                        <button className="btn-primary" onClick={handleProceed}
                            style={{ fontSize: '1.1rem', padding: '16px 48px' }}>
                            🚀 Start Analysis Pipeline
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}
