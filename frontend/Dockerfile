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

# Copy nginx config as a template — nginx's docker-entrypoint.sh automatically
# runs envsubst on files in /etc/nginx/templates/ before starting nginx.
# This lets Cloud Run inject PORT (and any other env vars) at container startup.
COPY nginx.conf /etc/nginx/templates/default.conf.template

# Cloud Run routes to whatever PORT is set (default 8080 if --port not given).
# The template substitution handles whichever value Cloud Run injects.
EXPOSE 8080

CMD ["nginx", "-g", "daemon off;"]
