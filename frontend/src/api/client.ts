import axios from 'axios';

const API_BASE = import.meta.env.VITE_ANALYTICS_API || '/api';

const api = axios.create({
    baseURL: API_BASE,
    timeout: 60000,
    headers: { 'Content-Type': 'application/json' },
});

const inFlightTeamColorRequests = new Map<number, Promise<any>>();

export const buildApiUrl = (path: string) =>
    `${API_BASE.replace(/\/$/, '')}${path.startsWith('/') ? path : `/${path}`}`;

// ── Upload ────────────────────────────────────────────────────────────────
export type UploadCommentaryOptions = {
    commentaryLevel: 'Auto' | 'Beginner' | 'Intermediate' | 'Expert',
    commentaryVerbosity?: 'low' | 'medium' | 'high',
    commentaryStyle?: 'neutral' | 'friendly' | 'analytical' | 'coach',
    educationalMode?: boolean,
    footballKnowledge?: '' | 'beginner' | 'intermediate' | 'expert',
    homeTeamName?: string,
    awayTeamName?: string,
};

export type FootballVideoValidation = {
    is_valid: boolean,
    status: 'accepted' | 'uncertain' | 'invalid' | 'skipped' | string,
    confidence: number,
    message: string,
    sampled_frames: number,
    positive_frame_ratio: number,
    evidence: Record<string, number>,
    frame_scores: Record<string, number>[],
};

export const validateFootballVideo = (file: File) =>
    api.post<FootballVideoValidation>('/validate-football-video', (() => {
        const f = new FormData();
        f.append('video', file);
        return f;
    })(), {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 0,
    });

export const uploadVideo = (
    file: File,
    options: UploadCommentaryOptions,
    onProgress?: (pct: number) => void,
) =>
    api.post('/upload-video', (() => {
        const f = new FormData();
        f.append('video', file);
        f.append('commentary_level', options.commentaryLevel);
        f.append('commentary_verbosity', options.commentaryVerbosity || 'medium');
        f.append('commentary_style', options.commentaryStyle || 'neutral');
        f.append('educational_mode', String(Boolean(options.educationalMode)));
        if (options.footballKnowledge) {
            f.append('football_knowledge', options.footballKnowledge);
        }
        return f;
    })(), {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 0,
        onUploadProgress: (e) => onProgress?.(Math.round((e.loaded * 100) / (e.total || 1))),
    });

export const submitLineups = (matchId: number, data: any) =>
    api.post(`/match/${matchId}/lineups`, data);

export const detectTeamColors = (matchId: number) => {
    const existing = inFlightTeamColorRequests.get(matchId);
    if (existing) {
        return existing;
    }

    const request = api
        .post(`/match/${matchId}/detect-team-colors`)
        .finally(() => {
            inFlightTeamColorRequests.delete(matchId);
        });

    inFlightTeamColorRequests.set(matchId, request);
    return request;
};

export const confirmTeamMapping = (matchId: number, teamNames: Record<number, string>) =>
    api.post(`/match/${matchId}/team-mapping`, { team_names: teamNames });

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

// ── Decision Quality ──────────────────────────────────────────────────────
export const getDecisionQuality = (matchId: number) =>
    api.get(`/match/${matchId}/decision-quality`);

// ── Analysis ──────────────────────────────────────────────────────────────
export const getAiAnalysis = (matchId: number) => api.post(`/match/${matchId}/ai-analysis`);

// ── Health ────────────────────────────────────────────────────────────────
export const getHealth = () => api.get('/health');

export default api;
