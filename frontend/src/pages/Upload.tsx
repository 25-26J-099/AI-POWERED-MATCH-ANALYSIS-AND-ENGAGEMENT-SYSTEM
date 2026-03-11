import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { uploadVideo } from '../api/client';

export default function Upload() {
    const navigate = useNavigate();
    const [file, setFile] = useState<File | null>(null);
    const [progress, setProgress] = useState(0);
    const [uploading, setUploading] = useState(false);
    const [error, setError] = useState('');
    const [dragOver, setDragOver] = useState(false);

    const handleFile = (candidate: File | null | undefined) => {
        if (!candidate) {
            return;
        }
        if (!candidate.type.startsWith('video/')) {
            setError('Please upload a video file.');
            return;
        }
        setFile(candidate);
        setError('');
    };

    const handleUpload = async () => {
        if (!file) {
            return;
        }
        setUploading(true);
        setError('');
        try {
            const response = await uploadVideo(file, setProgress);
            navigate(`/lineup/${response.data.match_id}`);
        } catch (err: any) {
            setError(err.response?.data?.detail || 'Upload failed');
        } finally {
            setUploading(false);
        }
    };

    return (
        <div className="page-container" style={{ maxWidth: '900px', margin: '0 auto', paddingTop: '48px' }}>
            <h1 className="page-title">Upload Match Video</h1>
            <p className="page-subtitle">Upload once, then track the merged tracking and analytics pipeline in real time.</p>

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

            <div
                className="glass-card"
                onDrop={(e) => {
                    e.preventDefault();
                    setDragOver(false);
                    handleFile(e.dataTransfer.files?.[0]);
                }}
                onDragOver={(e) => {
                    e.preventDefault();
                    setDragOver(true);
                }}
                onDragLeave={() => setDragOver(false)}
                onClick={() => document.getElementById('video-input')?.click()}
                style={{
                    border: dragOver ? '2px dashed var(--accent)' : '2px dashed var(--border-subtle)',
                    textAlign: 'center',
                    padding: '72px 40px',
                    cursor: 'pointer',
                    background: dragOver ? 'rgba(99,102,241,0.05)' : 'var(--bg-glass)',
                    transition: 'all 0.3s',
                }}
            >
                <div style={{ fontSize: '4rem', marginBottom: '16px' }}>🎬</div>
                <h3 style={{ fontSize: '1.3rem', fontWeight: 600, marginBottom: '8px' }}>
                    {file ? file.name : 'Drop your match video here'}
                </h3>
                <p style={{ color: 'var(--text-secondary)' }}>
                    {file ? `${(file.size / 1024 / 1024).toFixed(1)} MB ready to process` : 'or click to browse • MP4, AVI, MOV'}
                </p>
                <input
                    id="video-input"
                    type="file"
                    accept="video/*"
                    style={{ display: 'none' }}
                    onChange={(e) => handleFile(e.target.files?.[0])}
                />
            </div>

            {file && (
                <div style={{ marginTop: '24px' }}>
                    {uploading && (
                        <div style={{ marginBottom: '16px' }}>
                            <div
                                style={{
                                    height: '8px',
                                    borderRadius: '4px',
                                    background: 'var(--bg-card)',
                                    overflow: 'hidden',
                                }}
                            >
                                <div
                                    style={{
                                        height: '100%',
                                        width: `${progress}%`,
                                        background: 'var(--gradient-primary)',
                                        borderRadius: '4px',
                                        transition: 'width 0.3s',
                                    }}
                                />
                            </div>
                            <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: '8px' }}>
                                Uploading... {progress}%
                            </p>
                        </div>
                    )}
                    <button
                        className="btn-primary"
                        onClick={handleUpload}
                        disabled={uploading}
                        style={{ width: '100%', padding: '16px' }}
                    >
                        {uploading ? 'Uploading...' : 'Upload and Start Analysis →'}
                    </button>
                </div>
            )}
        </div>
    );
}
