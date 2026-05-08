import { useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { uploadVideo, validateFootballVideo } from '../api/client';

const COMMENTARY_LEVELS = ['Auto', 'Beginner', 'Intermediate', 'Expert'] as const;
const COMMENTARY_VERBOSITY = ['low', 'medium', 'high'] as const;
const COMMENTARY_STYLES = ['neutral', 'friendly', 'analytical', 'coach'] as const;
const FOOTBALL_KNOWLEDGE = ['', 'beginner', 'intermediate', 'expert'] as const;

export default function Upload() {
    const navigate = useNavigate();
    const [file, setFile] = useState<File | null>(null);
    const [commentaryLevel, setCommentaryLevel] = useState<(typeof COMMENTARY_LEVELS)[number]>('Auto');
    const [commentaryVerbosity, setCommentaryVerbosity] = useState<(typeof COMMENTARY_VERBOSITY)[number]>('medium');
    const [commentaryStyle, setCommentaryStyle] = useState<(typeof COMMENTARY_STYLES)[number]>('neutral');
    const [footballKnowledge, setFootballKnowledge] = useState<(typeof FOOTBALL_KNOWLEDGE)[number]>('');
    const [educationalMode, setEducationalMode] = useState(false);
    const [progress, setProgress] = useState(0);
    const [uploading, setUploading] = useState(false);
    const [validatingFile, setValidatingFile] = useState(false);
    const [pendingFileName, setPendingFileName] = useState('');
    const [validationMessage, setValidationMessage] = useState('');
    const [validationStatus, setValidationStatus] = useState('');
    const [error, setError] = useState('');
    const [dragOver, setDragOver] = useState(false);
    const validationRun = useRef(0);

    const handleFile = async (candidate: File | null | undefined) => {
        if (!candidate) {
            return;
        }
        const runId = validationRun.current + 1;
        validationRun.current = runId;
        setFile(null);
        setProgress(0);
        setValidationMessage('');
        setValidationStatus('');

        if (!candidate.type.startsWith('video/')) {
            setError('Please upload a video file.');
            return;
        }

        setError('');
        setPendingFileName(candidate.name);
        setValidatingFile(true);
        setValidationStatus('checking');
        setValidationMessage('Checking whether this is football match footage...');

        try {
            const response = await validateFootballVideo(candidate);
            if (validationRun.current !== runId) {
                return;
            }
            const validation = response.data;
            setValidationStatus(validation.status);
            setValidationMessage(validation.message);
            if (validation.is_valid) {
                setFile(candidate);
            } else {
                setFile(null);
                setError(validation.message);
            }
        } catch (err: any) {
            if (validationRun.current !== runId) {
                return;
            }
            const detail = err.response?.data?.detail || 'Could not validate this video. Please try another file.';
            setFile(null);
            setValidationStatus('invalid');
            setValidationMessage(detail);
            setError(detail);
        } finally {
            if (validationRun.current === runId) {
                setValidatingFile(false);
            }
        }
    };

    const handleUpload = async () => {
        if (!file) {
            return;
        }
        setUploading(true);
        setError('');
        try {
            const response = await uploadVideo(file, {
                commentaryLevel,
                commentaryVerbosity,
                commentaryStyle,
                educationalMode,
                footballKnowledge,
            }, setProgress);
            navigate(`/teams/${response.data.match_id}`);
        } catch (err: any) {
            setError(err.response?.data?.detail || 'Upload failed');
        } finally {
            setUploading(false);
        }
    };

    return (
        <div className="page-container" style={{ maxWidth: '900px', margin: '0 auto', paddingTop: '48px' }}>
            <h1 className="page-title">Upload Match Video</h1>
            <p className="page-subtitle">Upload once, validate match footage, then track the merged analytics pipeline in real time.</p>

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
                    void handleFile(e.dataTransfer.files?.[0]);
                }}
                onDragOver={(e) => {
                    e.preventDefault();
                    setDragOver(true);
                }}
                onDragLeave={() => setDragOver(false)}
                onClick={() => {
                    if (!uploading && !validatingFile) {
                        const input = document.getElementById('video-input') as HTMLInputElement | null;
                        if (input) {
                            input.value = '';
                            input.click();
                        }
                    }
                }}
                style={{
                    border: dragOver ? '2px dashed var(--accent)' : '2px dashed var(--border-subtle)',
                    textAlign: 'center',
                    padding: '72px 40px',
                    cursor: uploading || validatingFile ? 'not-allowed' : 'pointer',
                    background: dragOver ? 'rgba(99,102,241,0.05)' : 'var(--bg-glass)',
                    transition: 'all 0.3s',
                }}
            >
                <div style={{ fontSize: '4rem', marginBottom: '16px' }}>🎬</div>
                <h3 style={{ fontSize: '1.3rem', fontWeight: 600, marginBottom: '8px' }}>
                    {validatingFile ? pendingFileName : file ? file.name : 'Drop your match video here'}
                </h3>
                <p style={{ color: 'var(--text-secondary)' }}>
                    {validatingFile
                        ? 'Checking football match content before upload...'
                        : file
                            ? `${(file.size / 1024 / 1024).toFixed(1)} MB ready to process`
                            : 'or click to browse - MP4, AVI, MOV'}
                </p>
                <input
                    id="video-input"
                    type="file"
                    accept="video/*"
                    style={{ display: 'none' }}
                    disabled={uploading || validatingFile}
                    onChange={(e) => void handleFile(e.target.files?.[0])}
                />
            </div>

            {validationMessage && (
                <div
                    style={{
                        background:
                            validationStatus === 'invalid'
                                ? 'rgba(239,68,68,0.1)'
                                : validationStatus === 'checking'
                                    ? 'rgba(59,130,246,0.1)'
                                    : 'rgba(34,197,94,0.1)',
                        border:
                            validationStatus === 'invalid'
                                ? '1px solid rgba(239,68,68,0.3)'
                                : validationStatus === 'checking'
                                    ? '1px solid rgba(59,130,246,0.3)'
                                    : '1px solid rgba(34,197,94,0.3)',
                        borderRadius: '12px',
                        padding: '12px 16px',
                        marginTop: '16px',
                        color:
                            validationStatus === 'invalid'
                                ? '#ef4444'
                                : validationStatus === 'checking'
                                    ? '#60a5fa'
                                    : '#22c55e',
                    }}
                >
                    {validationMessage}
                </div>
            )}

            <div className="glass-card" style={{ marginTop: '24px' }}>
                <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '10px' }}>Expert Commentary Level</h3>
                <p style={{ color: 'var(--text-secondary)', marginBottom: '16px', lineHeight: 1.6 }}>
                    This changes only the tactical expert commentary. Choose `Auto` to let the learned audience model infer the best level from your preferences.
                </p>
                <div style={{ display: 'grid', gap: '12px', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))' }}>
                    {COMMENTARY_LEVELS.map((level) => {
                        const selected = commentaryLevel === level;
                        return (
                            <button
                                key={level}
                                type="button"
                                onClick={() => setCommentaryLevel(level)}
                                disabled={uploading}
                                style={{
                                    borderRadius: '14px',
                                    border: selected ? '1px solid var(--accent)' : '1px solid var(--border-subtle)',
                                    background: selected ? 'rgba(99,102,241,0.12)' : 'var(--bg-glass)',
                                    padding: '16px',
                                    textAlign: 'left',
                                    cursor: uploading ? 'not-allowed' : 'pointer',
                                    opacity: uploading ? 0.6 : 1,
                                    transition: 'all 0.2s ease',
                                }}
                            >
                                <div style={{ fontWeight: 700, marginBottom: '6px', color: 'var(--text-primary)' }}>{level}</div>
                                <div style={{ color: 'var(--text-secondary)', fontSize: '0.95rem', lineHeight: 1.5 }}>
                                    {level === 'Auto' && 'Uses learned audience modeling to infer the tactical commentary level from your viewing preferences.'}
                                    {level === 'Beginner' && 'Simple tactical explanations with clear, accessible language.'}
                                    {level === 'Intermediate' && 'Balanced tactical detail with readable match analysis.'}
                                    {level === 'Expert' && 'Dense tactical language aimed at advanced football analysis.'}
                                </div>
                            </button>
                        );
                    })}
                </div>
            </div>

            <div className="glass-card" style={{ marginTop: '24px' }}>
                <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '10px' }}>Audience Modeling Preferences</h3>
                <p style={{ color: 'var(--text-secondary)', marginBottom: '16px', lineHeight: 1.6 }}>
                    These preferences help the learned audience model tailor the tactical commentary when `Auto` is selected, and they still guide style when a fixed level is chosen.
                </p>

                <div style={{ display: 'grid', gap: '16px', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))' }}>
                    <label style={{ display: 'grid', gap: '8px' }}>
                        <span style={{ fontWeight: 600 }}>Football knowledge</span>
                        <select
                            value={footballKnowledge}
                            onChange={(e) => setFootballKnowledge(e.target.value as (typeof FOOTBALL_KNOWLEDGE)[number])}
                            disabled={uploading}
                            style={{ borderRadius: '12px', padding: '12px', background: 'var(--bg-card)', color: 'var(--text-primary)', border: '1px solid var(--border-subtle)' }}
                        >
                            <option value="">Auto detect from other signals</option>
                            <option value="beginner">Beginner</option>
                            <option value="intermediate">Intermediate</option>
                            <option value="expert">Expert</option>
                        </select>
                    </label>

                    <label style={{ display: 'grid', gap: '8px' }}>
                        <span style={{ fontWeight: 600 }}>Preferred detail</span>
                        <select
                            value={commentaryVerbosity}
                            onChange={(e) => setCommentaryVerbosity(e.target.value as (typeof COMMENTARY_VERBOSITY)[number])}
                            disabled={uploading}
                            style={{ borderRadius: '12px', padding: '12px', background: 'var(--bg-card)', color: 'var(--text-primary)', border: '1px solid var(--border-subtle)' }}
                        >
                            {COMMENTARY_VERBOSITY.map((option) => (
                                <option key={option} value={option}>{option}</option>
                            ))}
                        </select>
                    </label>

                    <label style={{ display: 'grid', gap: '8px' }}>
                        <span style={{ fontWeight: 600 }}>Commentary style</span>
                        <select
                            value={commentaryStyle}
                            onChange={(e) => setCommentaryStyle(e.target.value as (typeof COMMENTARY_STYLES)[number])}
                            disabled={uploading}
                            style={{ borderRadius: '12px', padding: '12px', background: 'var(--bg-card)', color: 'var(--text-primary)', border: '1px solid var(--border-subtle)' }}
                        >
                            {COMMENTARY_STYLES.map((option) => (
                                <option key={option} value={option}>{option}</option>
                            ))}
                        </select>
                    </label>
                </div>

                <label style={{ display: 'flex', alignItems: 'center', gap: '10px', marginTop: '16px', color: 'var(--text-secondary)' }}>
                    <input
                        type="checkbox"
                        checked={educationalMode}
                        onChange={(e) => setEducationalMode(e.target.checked)}
                        disabled={uploading}
                    />
                    Prefer teaching-style tactical explanations
                </label>
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
                                {progress >= 100 ? 'Validating football match content...' : `Uploading... ${progress}%`}
                            </p>
                        </div>
                    )}

                    <button
                        className="btn-primary"
                        onClick={handleUpload}
                        disabled={uploading || validatingFile || !file}
                        style={{ width: '100%', padding: '16px' }}
                    >
                        {uploading ? 'Uploading...' : `Upload and Start Analysis (${commentaryLevel}) ->`}
                    </button>
                </div>
            )}
        </div>
    );
}
