#!/usr/bin/env bash
# =============================================================================
# GCE VM startup script — runs once on first boot after VM creation.
# Installs NVIDIA drivers, Docker, nvidia-container-toolkit, and starts
# the full docker-compose stack.
#
# This script is passed to the VM via --metadata-from-file=startup-script.
# Logs are available at: /var/log/startup-script.log
# =============================================================================

set -euo pipefail
exec > /var/log/startup-script.log 2>&1

echo "=== VM startup script started at $(date) ==="

PROJECT_ID=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/project/project-id" \
  -H "Metadata-Flavor: Google")
GCS_BUCKET=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/attributes/GCS_BUCKET" \
  -H "Metadata-Flavor: Google" || echo "")
SQL_CONNECTION=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/attributes/SQL_CONNECTION" \
  -H "Metadata-Flavor: Google" || echo "")
REGION="us-central1"
AR_REPO="football-ai"

# ── 1. NVIDIA CUDA drivers ────────────────────────────────────────────────────
if ! command -v nvidia-smi &>/dev/null; then
  echo ">>> Installing NVIDIA drivers..."
  apt-get update -q
  apt-get install -y --no-install-recommends \
    linux-headers-$(uname -r) \
    build-essential \
    dkms

  # Install CUDA keyring
  curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    -o /tmp/cuda-keyring.deb
  dpkg -i /tmp/cuda-keyring.deb
  apt-get update -q
  apt-get install -y --no-install-recommends cuda-drivers
  echo ">>> NVIDIA drivers installed. A reboot may be required for first-time setup."
else
  echo ">>> NVIDIA drivers already installed: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
fi

# ── 2. Docker ─────────────────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
  echo ">>> Installing Docker..."
  curl -fsSL https://get.docker.com | sh
  systemctl enable docker
  systemctl start docker
else
  echo ">>> Docker already installed: $(docker --version)"
fi

# ── 3. NVIDIA Container Toolkit ───────────────────────────────────────────────
if ! dpkg -l | grep -q nvidia-container-toolkit; then
  echo ">>> Installing nvidia-container-toolkit..."
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -q
  apt-get install -y nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
else
  echo ">>> nvidia-container-toolkit already installed"
fi

# ── 4. Configure Docker to authenticate with Artifact Registry ────────────────
echo ">>> Configuring Docker auth for Artifact Registry..."
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# ── 5. Clone/update the application repository ───────────────────────────────
APP_DIR="/app"
REPO_URL=$(curl -sf \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/REPO_URL" \
  -H "Metadata-Flavor: Google" || echo "")

if [ -d "${APP_DIR}/.git" ]; then
  echo ">>> Updating application repository..."
  git -C "${APP_DIR}" pull --ff-only
elif [ -n "${REPO_URL}" ]; then
  echo ">>> Cloning application repository..."
  git clone "${REPO_URL}" "${APP_DIR}"
else
  echo ">>> Creating app directory (repo URL not set — deploy via Cloud Build)..."
  mkdir -p "${APP_DIR}"
fi

cd "${APP_DIR}"

# Ensure backend dir exists (repo not cloned yet on first boot)
mkdir -p "${APP_DIR}/backend"

# ── 6. Write .env for docker-compose (secrets pulled from Secret Manager) ─────
echo ">>> Fetching secrets from Secret Manager..."
DB_PASSWORD=$(gcloud secrets versions access latest --secret=DB_PASSWORD 2>/dev/null || echo "devpassword")
DATABASE_URL=$(gcloud secrets versions access latest --secret=DATABASE_URL 2>/dev/null || echo "")
OPENAI_API_KEY=$(gcloud secrets versions access latest --secret=OPENAI_API_KEY 2>/dev/null || echo "")
HF_TOKEN=$(gcloud secrets versions access latest --secret=HF_TOKEN 2>/dev/null || echo "")
PREFECT_API_KEY=$(gcloud secrets versions access latest --secret=PREFECT_API_KEY 2>/dev/null || echo "")
PREFECT_API_URL=$(gcloud secrets versions access latest --secret=PREFECT_API_URL 2>/dev/null || echo "")

# Write root .env (used by docker-compose.yml)
cat > "${APP_DIR}/.env" <<EOF
DB_PASSWORD=${DB_PASSWORD}
GCS_BUCKET=${GCS_BUCKET}
REGISTRY=${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}
GCP_KEY_PATH=
EOF

# Write backend .env (mounted via env_file in docker-compose)
cat > "${APP_DIR}/backend/.env" <<EOF
DATABASE_URL=${DATABASE_URL}
OPENAI_API_KEY=${OPENAI_API_KEY}
OPENAI_MODEL=gpt-4o
HF_FOOTBALL_MODELS_REPO=AI-POWERED-FOOTBALL-SYSTEM/football-ai-models
HF_XG_REPO=AI-POWERED-FOOTBALL-SYSTEM/football-ai-models
HF_VAEP_SCORING_REPO=AI-POWERED-FOOTBALL-SYSTEM/football-ai-models
HF_VAEP_CONCEDING_REPO=AI-POWERED-FOOTBALL-SYSTEM/football-ai-models
HF_STYLE_SCALER_REPO=AI-POWERED-FOOTBALL-SYSTEM/football-ai-models
HF_STYLE_AUTOENCODER_REPO=AI-POWERED-FOOTBALL-SYSTEM/football-ai-models
HF_STYLE_KMEANS_REPO=AI-POWERED-FOOTBALL-SYSTEM/football-ai-models
HF_TOKEN=${HF_TOKEN}
HF_CACHE_DIR=/data/model_cache
GCS_BUCKET=${GCS_BUCKET}
MLFLOW_TRACKING_URI=http://mlflow:5000
PREFECT_API_KEY=${PREFECT_API_KEY}
PREFECT_API_URL=${PREFECT_API_URL}
FORCE_CPU=false
GNN_DEVICE=cuda
FASTREID_ENABLED=true
TORCHREID_DEVICE=cuda
CORS_ORIGINS=["http://localhost","http://localhost:80","http://localhost:3000"]
EOF

# ── 7. Pull and start docker-compose stack (only if docker-compose.yml exists) ──
if [ -f "${APP_DIR}/docker-compose.yml" ]; then
  echo ">>> Pulling Docker images..."
  docker compose pull --quiet 2>/dev/null || echo "    (some images not yet in registry — will be deployed via Cloud Build)"

  echo ">>> Starting services..."
  docker compose up -d --remove-orphans || echo "    (startup failed — check docker compose logs)"

  echo ""
  echo "=== VM startup complete at $(date) ==="
  echo "    Services: $(docker compose ps --services)"
else
  echo ">>> No docker-compose.yml found — first boot setup complete."
  echo ">>> Cloud Build will deploy the application on next git push to main."
  echo ""
  echo "=== VM startup complete at $(date) ==="
fi
