# FastReID Assets

This backend now resolves FastReID from project-local paths by default.

Required files:

- `backend/models/reid/fastreid/configs/football_vit.yml`
- `backend/models/reid/fastreid/weights/football_vit.pth`

The config file is already included in this repository. You still need to add a compatible FastReID checkpoint at:

- `backend/models/reid/fastreid/weights/football_vit.pth`

Environment overrides supported through `.env`:

- `FASTREID_CONFIG_PATH`
- `FASTREID_WEIGHTS_PATH`
- `FASTREID_STRICT`
- `FASTREID_DEVICE`
- `HF_FASTREID_REPO`
- `HF_FASTREID_CONFIG_FILE`
- `HF_FASTREID_WEIGHTS_FILE`

If `HF_FASTREID_REPO` is configured, the backend will try to download missing FastReID assets from Hugging Face before falling back.

Recommended Windows verification command:

```powershell
python scripts/check_reid_backend.py
```
