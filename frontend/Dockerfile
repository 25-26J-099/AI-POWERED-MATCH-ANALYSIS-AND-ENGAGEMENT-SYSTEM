# ── Frontend Dockerfile (multi-stage) ────────────────────────────────────────
# Stage 1: TypeScript compile + Vite bundle
# Stage 2: Serve with nginx (tiny alpine image)

# ── Stage 1: Build ────────────────────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

# Install dependencies (cached layer — only re-runs when package-lock changes)
COPY package*.json ./
RUN npm ci

# Copy source and build
COPY . .

# The backend API is proxied through nginx at /api — no CORS, no hardcoded host.
ENV VITE_ANALYTICS_API=/api

RUN npm run build


# ── Stage 2: Serve ────────────────────────────────────────────────────────────
FROM nginx:1.27-alpine

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Serve built frontend
COPY --from=builder /app/dist /usr/share/nginx/html

# Custom nginx config (SPA routing + /api proxy)
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
