# Backend Overview

This backend is the match-analysis and orchestration layer for the project. Its primary job is to convert low-quality single-camera football footage into structured match intelligence that downstream analytics and commentary components can use.

## What This Component Owns

- Video preprocessing for unstable, low-quality footage
- Player and ball tracking
- Team assignment and player identity continuity
- Hybrid event detection with rule-based and ML-assisted paths
- StatsBomb-style export and analytics handoff
- Tactical-analysis support used by later commentary stages

In project-report terms, this backend is the analysis engine that powers the other components rather than the final commentary experience itself.

## Core Entry Points

- API app: `app/main.py`
- Video analysis API: `app/routes/video.py`
- Match-analysis pipeline: `app/event_detection/pipeline.py`
- Preprocessing: `app/event_detection/video_preprocessor.py`
- Tracking: `app/event_detection/tracker.py`
- Re-identification: `app/event_detection/player_reid.py`
- Hybrid event detection: `app/event_detection/strategic_hybrid_detector.py`
- Tactical GNN support: `app/tactical_gnn/README.md`

## Quick Start

From the `backend` directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-windows.txt
uvicorn app.main:app --reload
```

On macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-mac.txt
uvicorn app.main:app --reload
```

Health check:

```powershell
curl http://127.0.0.1:8000/health
```

## Reproducibility Notes

- `pydantic-settings` is required because `app/config/settings.py` imports `BaseSettings` from that package.
- Some tests and runtime paths also require heavier ML dependencies such as `joblib`, PyTorch, and OpenCV extras.
- Model-backed flows may attempt to load local or cached assets from Hugging Face through the repo's model loader.
- The Re-ID layer now resolves backends in this order: `FastReID` -> `torchreid` -> handcrafted fallback.

## Re-ID Backends

The live stable-ID mapping is handled by `app/event_detection/robust_reid.py`, which uses the shared `app/event_detection/reid_module.py` backend ladder.

- Preferred: `FastReID` with a ViT-compatible config and checkpoint
- Practical fallback: `torchreid` with `osnet_ain_x1_0`
- Safe fallback: handcrafted embeddings when deep backends are unavailable

FastReID is now wired to repo-local defaults:

- Config: `models/reid/fastreid/configs/football_vit.yml`
- Weights: `models/reid/fastreid/weights/football_vit.pth`

To override them, use `.env` values:

- `FASTREID_CONFIG_PATH`
- `FASTREID_WEIGHTS_PATH`
- `FASTREID_DEVICE`
- `FASTREID_STRICT`
- `HF_FASTREID_REPO`

If FastReID is not installed or not configured, the pipeline automatically falls back to `torchreid`. If neither deep backend is available, the pipeline still runs with the handcrafted backend.

## Windows FastReID Setup

Use the backend virtual environment explicitly so the runtime matches the installed packages:

```powershell
cd backend
.\venv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable)"
```

If your active interpreter does not point into `backend\venv` or `.venv`, fix that before running the pipeline.

The Windows Re-ID extras are included in `requirements-windows.txt`.

If CUDA is unavailable on Windows, keep `FASTREID_DEVICE=cpu` or leave it as `auto`.

This repo now includes Python 3.12 compatibility shims for legacy FastReID imports:

- `collections.Mapping` and related aliases
- `torch._six`

Those shims are loaded automatically from [sitecustomize.py](d:/AI-POWERED-MATCH-ANALYSIS-AND-ENGAGEMENT-SYSTEM-Dev2-PP2-Sachintha_2/backend/sitecustomize.py), so you do not need to patch the installed package manually.

After adding the FastReID checkpoint file, run:

```powershell
python scripts/check_reid_backend.py
```

That script reports:

- the active Python interpreter
- whether `fastreid`, `torch`, `torchvision`, `cv2`, and `torchreid` can be imported
- whether the config and weight files exist
- which Re-ID backend the backend will actually use

## Suggested Verification Flow

Run lightweight API and tactical tests first:

```powershell
pytest tests/api/test_video_router.py -q
pytest tests/tactical_gnn/test_graph_builder.py -q
```

Then run the broader tactical and service suites:

```powershell
pytest tests/tactical_gnn tests/integration/test_tactical_gnn_integration.py tests/services/test_statsbomb_export.py -q
```

## Research-Safe Positioning

The backend can honestly claim:

- An end-to-end analysis pipeline exists for low-resource football footage
- Structured event export and analytics handoff are implemented
- Tactical GNN training, inference, and qualitative comparison tooling exist
- Heuristic fallback paths are preserved when ML components are unavailable

The backend should not over-claim:

- Robust real-match tactical performance without real labeled evaluation
- Complete validation of all proposal-level low-resource and real-time claims without benchmark evidence
- A published real amateur-football dataset unless that dataset is actually packaged and released

## Supporting Documents

- Individual contribution report: `docs/INDIVIDUAL_CONTRIBUTION_REPORT.md`
- Evaluation template for evidence collection: `docs/LOW_RESOURCE_EVALUATION_REPORT.md`
