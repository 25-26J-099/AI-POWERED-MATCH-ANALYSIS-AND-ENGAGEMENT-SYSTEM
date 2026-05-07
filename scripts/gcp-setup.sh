#!/usr/bin/env bash
# =============================================================================
# GCP one-time setup script for the AI Football Analysis System
# Run this ONCE. Project football-ai-495521 must already exist with billing linked.
#
# Prerequisites:
#   1. gcloud CLI authenticated: gcloud auth login
#   2. Run: gcloud config set project football-ai-495521
#
# Usage:
#   chmod +x scripts/gcp-setup.sh
#   bash scripts/gcp-setup.sh
# =============================================================================

set -euo pipefail

# Hard-code the project so the script is safe to re-run without needing
# the caller to have set gcloud config first.
PROJECT_ID="football-ai-495521"

# Use Homebrew gcloud installation
GCLOUD="/opt/homebrew/bin/gcloud"
GSUTIL="/opt/homebrew/bin/gsutil"

$GCLOUD config set project "${PROJECT_ID}" --quiet
REGION="us-central1"
ZONE="us-east1-b"
BUCKET="football-ai-495521"
AR_REPO="football-ai"
SQL_INSTANCE="football-ai-db"
VM_NAME="football-ai-backend-new"
SA_NAME="football-ai-sa"

echo "=== Setting up GCP project: ${PROJECT_ID} ==="
echo "    Region: ${REGION} | Zone: ${ZONE}"
echo ""

# ── 1. Enable required APIs ───────────────────────────────────────────────────
echo ">>> Enabling GCP APIs..."
$GCLOUD services enable \
  run.googleapis.com \
  compute.googleapis.com \
  sqladmin.googleapis.com \
  storage.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com \
  iam.googleapis.com \
  oslogin.googleapis.com

# ── 2. Artifact Registry ──────────────────────────────────────────────────────
echo ">>> Creating Artifact Registry repository..."
$GCLOUD artifacts repositories create "${AR_REPO}" \
  --repository-format=docker \
  --location="${REGION}" \
  --description="Football AI Docker images" \
  2>/dev/null || echo "    (already exists)"

# ── 3. GCS Bucket ─────────────────────────────────────────────────────────────
echo ">>> Creating GCS bucket gs://${BUCKET} ..."
$GSUTIL mb -l "${REGION}" "gs://${BUCKET}" 2>/dev/null || echo "    (already exists)"

# Create lifecycle rule — delete incomplete multipart uploads after 7 days
cat > /tmp/lifecycle.json <<'EOF'
{
  "lifecycle": {
    "rule": [{
      "action": {"type": "AbortIncompleteMultipartUpload"},
      "condition": {"age": 7}
    }]
  }
}
EOF
$GSUTIL lifecycle set /tmp/lifecycle.json "gs://${BUCKET}"

# ── 4. Cloud SQL (PostgreSQL 15) ───────────────────────────────────────────────
echo ">>> Creating Cloud SQL instance (this takes ~5 minutes)..."
$GCLOUD sql instances create "${SQL_INSTANCE}" \
  --database-version=POSTGRES_15 \
  --tier=db-g1-small \
  --region="${REGION}" \
  --storage-type=SSD \
  --storage-size=20GB \
  --backup-start-time=03:00 \
  --availability-type=zonal \
  2>/dev/null || echo "    (already exists)"

echo ">>> Creating databases and user..."
# Generate a random DB password
DB_PASSWORD=$(openssl rand -base64 24 | tr -dc 'A-Za-z0-9' | head -c 32)

$GCLOUD sql users create appuser \
  --instance="${SQL_INSTANCE}" \
  --password="${DB_PASSWORD}" \
  2>/dev/null || echo "    (user already exists — password NOT reset)"

$GCLOUD sql databases create football_analysis \
  --instance="${SQL_INSTANCE}" \
  2>/dev/null || echo "    (football_analysis db already exists)"

$GCLOUD sql databases create mlflow \
  --instance="${SQL_INSTANCE}" \
  2>/dev/null || echo "    (mlflow db already exists)"

SQL_CONNECTION=$($GCLOUD sql instances describe "${SQL_INSTANCE}" \
  --format='value(connectionName)')

echo "    SQL connection name: ${SQL_CONNECTION}"

# ── 5. Service Account ────────────────────────────────────────────────────────
echo ">>> Creating service account ${SA_NAME}..."
$GCLOUD iam service-accounts create "${SA_NAME}" \
  --display-name="Football AI Backend SA" \
  2>/dev/null || echo "    (already exists)"

SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

for role in \
  roles/storage.objectAdmin \
  roles/cloudsql.client \
  roles/secretmanager.secretAccessor \
  roles/artifactregistry.reader; do
  $GCLOUD projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="${role}" \
    --quiet
done

# ── 6. Secret Manager — store .env secrets ────────────────────────────────────
echo ">>> Storing secrets in Secret Manager..."

store_secret() {
  local name="$1"
  local value="$2"
  echo -n "${value}" | $GCLOUD secrets create "${name}" \
    --data-file=- \
    --replication-policy=automatic \
    2>/dev/null || \
  echo -n "${value}" | $GCLOUD secrets versions add "${name}" --data-file=-
}

store_secret "DB_PASSWORD" "${DB_PASSWORD}"
store_secret "DATABASE_URL" \
  "postgresql+asyncpg://appuser:${DB_PASSWORD}@/football_analysis?host=/cloudsql/${SQL_CONNECTION}"

# Prefect Cloud API key (provided by user)
store_secret "PREFECT_API_KEY" "pnu_5cAWAtRb9MSUkvJ3caENOTf2RYwLYe0HkN7L"
# Prefect API URL will be set after you create a workspace at https://app.prefect.cloud
# Run: $GCLOUD secrets create PREFECT_API_URL --data-file=- <<< 'https://api.prefect.cloud/api/accounts/ACCOUNT_ID/workspaces/WORKSPACE_ID' 

echo ""
echo "=== IMPORTANT — manual secrets to add ==================================="
echo "    Run the following to add secrets that require your personal keys:"
echo ""
echo "    $GCLOUD secrets create OPENAI_API_KEY --data-file=- <<< 'sk-...'"
echo "    $GCLOUD secrets create HF_TOKEN        --data-file=- <<< 'hf_...'"
echo "    $GCLOUD secrets create PREFECT_API_KEY --data-file=- <<< 'pnu_...'"
echo "    $GCLOUD secrets create PREFECT_API_URL --data-file=- <<< 'https://api.prefect.cloud/...'"
echo "========================================================================="
echo ""

# ── 7. GCE GPU VM (NVIDIA L4) ──────────────────────────────────────────────────
echo ">>> Creating GCE VM ${VM_NAME} (g2-standard-32 · 1x NVIDIA L4 24GB · ~\$1.49/hr)..."
echo "    This may take 2–3 minutes..."

$GCLOUD compute instances create "${VM_NAME}" \
  --zone="${ZONE}" \
  --machine-type=g2-standard-8 \
  --maintenance-policy=TERMINATE \
  --restart-on-failure \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-ssd \
  --boot-disk-device-name="${VM_NAME}-disk" \
  --service-account="${SA_EMAIL}" \
  --scopes=cloud-platform \
  --tags=http-server,https-server \
  --metadata-from-file=startup-script=scripts/vm-startup.sh \
  --metadata="GCS_BUCKET=${BUCKET},SQL_CONNECTION=${SQL_CONNECTION}" \
  2>/dev/null || echo "    (VM already exists)"

# ── 8. Firewall rules ─────────────────────────────────────────────────────────
echo ">>> Creating firewall rules..."
$GCLOUD compute firewall-rules create allow-football-ai \
  --allow=tcp:8000,tcp:5001 \
  --target-tags=http-server \
  --description="Allow backend (8000) and MLflow (5001) traffic" \
  2>/dev/null || echo "    (firewall rule already exists)"

# ── 9. Cloud Build — grant SSH access to VM ───────────────────────────────────
echo ">>> Granting Cloud Build SA OS Login access to VM..."
CB_SA="$($GCLOUD projects describe ${PROJECT_ID} \
  --format='value(projectNumber)')@cloudbuild.gserviceaccount.com"

$GCLOUD projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${CB_SA}" \
  --role=roles/compute.osLogin --quiet

$GCLOUD projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${CB_SA}" \
  --role=roles/iam.serviceAccountUser --quiet

# ── 10. Configure Docker auth for Artifact Registry ──────────────────────────
echo ">>> Configuring Docker authentication for Artifact Registry..."
$GCLOUD auth configure-docker "${REGION}-docker.pkg.dev" --quiet

VM_IP=$($GCLOUD compute instances describe "${VM_NAME}" \
  --zone="${ZONE}" \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

echo ""
echo "=== Setup complete! ====================================================="
echo ""
echo "  VM external IP:   ${VM_IP}"
echo "  MLflow UI:        http://${VM_IP}:5001"
echo "  Backend API:      http://${VM_IP}:8000"
echo "  Artifact Registry: ${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/"
echo ""
echo "  Next steps:"
echo "  1. Add your personal secrets (OPENAI_API_KEY, HF_TOKEN, etc.) — see above"
echo "  2. Connect GitHub repo in Cloud Build Console:"
echo "     https://console.cloud.google.com/cloud-build/triggers"
echo "  3. Create trigger: push to main → cloudbuild.yaml"
echo "  4. Create Prefect Cloud account at https://prefect.io and store keys"
echo "  5. Wait ~5 min for VM startup script to finish, then:"
echo "     $GCLOUD compute ssh ${VM_NAME} --zone=${ZONE}"
echo "     docker compose ps"
echo "========================================================================"
