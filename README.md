# Hair AI — Ahmad idea (workspace)

Monorepo for hairstyle / portrait experiments.

## Contents

| Folder | Description |
|--------|-------------|
| **`AI-Hairstyle-TryOn/`** | Main app: FastAPI + web UI, rembg background removal, optional BiSeNet + LaMa hair removal & try-on. See its `README.md`. Run: `python run.py` from that folder. |
| **`hair_mesh_fit/`** | Utility script to fit hair mesh to head (`fit_hair_to_head.py`). |
| **`curly-hair-1-beatrice-character-hair-free/`** | Beatrice hair asset pack (large `.zip` files are gitignored by default). |

Root-level files include diagrams, sample images, and a project notebook.

## Quick start (main app)

```bash
cd AI-Hairstyle-TryOn
pip install -r requirements.txt
python run.py
```

Open the URL printed in the terminal (e.g. `http://127.0.0.1:8000/`).

## Notes

- **Weights** (LaMa, BiSeNet, etc.) download on first run into `AI-Hairstyle-TryOn/models/weights/` (ignored by git).
- Set **`SKIP_HAIR_PIPELINE=0`** for full hair pipeline; default fast mode is rembg-only.
