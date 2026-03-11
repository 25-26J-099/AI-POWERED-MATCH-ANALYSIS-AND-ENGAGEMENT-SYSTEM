import axios from 'axios';

const API_BASE = import.meta.env.VITE_ANALYTICS_API || '/api';

const api = axios.create({
    baseURL: API_BASE,
    timeout: 60000,
    headers: { 'Content-Type': 'application/json' },
});

export const buildApiUrl = (path: string) =>
    `${API_BASE.replace(/\/$/, '')}${path.startsWith('/') ? path : `/${path}`}`;

// ── Upload ────────────────────────────────────────────────────────────────
export const uploadVideo = (file: File, onProgress?: (pct: number) => void) =>
    api.post('/upload-video', (() => { const f = new FormData(); f.append('video', file); return f; })(), {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (e) => onProgress?.(Math.round((e.loaded * 100) / (e.total || 1))),
    });

export const submitLineups = (matchId: number, data: any) =>
    api.post(`/match/${matchId}/lineups`, data);

export const proceedPipeline = (matchId: number) =>
    api.post(`/match/${matchId}/proceed`);

// ── Match ─────────────────────────────────────────────────────────────────
export const getMatches = () => api.get('/matches');
export const getMatch = (id: number) => api.get(`/match/${id}`);
export const getMatchStatus = (id: number) => api.get(`/match/${id}/status`);
export const getMatchEvents = (id: number) => api.get(`/match/${id}/events`);
export const getMatchAnalytics = (id: number) => api.get(`/match/${id}/analytics`);
export const getMatchCommentary = (id: number) => api.get(`/match/${id}/commentary`);
export const getCommentaryVideo = (id: number) => api.get(`/match/${id}/commentary/video`);

// ── Players ───────────────────────────────────────────────────────────────
export const getMatchPlayers = (id: number) => api.get(`/match/${id}/players`);
export const getPlayerDetail = (matchId: number, playerId: number) =>
    api.get(`/match/${matchId}/player/${playerId}`);
export const getPlayerHeatmap = (matchId: number, playerId: number) =>
    api.get(`/match/${matchId}/player/${playerId}/heatmap`);
export const comparePlayersApi = (matchId: number, p1: number, p2: number) =>
    api.get(`/match/${matchId}/player-comparison`, { params: { player1: p1, player2: p2 } });

// ── Lineups ───────────────────────────────────────────────────────────────
export const getLineups = (matchId: number) => api.get(`/match/${matchId}/lineups`);

// ── Embeddings ────────────────────────────────────────────────────────────
export const getStyleMap = (matchId: number) => api.get(`/match/${matchId}/style-map`);

// ── Analysis ──────────────────────────────────────────────────────────────
export const getAiAnalysis = (matchId: number) => api.post(`/match/${matchId}/ai-analysis`);

// ── Health ────────────────────────────────────────────────────────────────
export const getHealth = () => api.get('/health');

export default api;
