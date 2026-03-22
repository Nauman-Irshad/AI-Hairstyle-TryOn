# AI Hairstyle Try-On System

**Try-on path (default):** **face landmarks** estimate the **hairline** (scalp band) → **BiSeNet** hair segmentation is combined with that region → **LaMa** inpaints your natural hair away → **hairstyle PNG** is warped (perspective/affine) and **alpha-blended** on the inpainted head. The API is **FastAPI + Uvicorn**; the UI is **HTML/CSS/JS**.

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA (recommended) — PyTorch uses `torch.device("cuda")` when available
- ~2 GB free disk for first-time downloads (BiSeNet + LaMa checkpoints)

## Setup

```bash
cd AI-Hairstyle-TryOn
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Generate default hairstyle PNG assets (if not already present):

```bash
python scripts/create_default_hairstyles.py
```

Optional dataset / CSV metadata:

```bash
python scripts/download_datasets.py
```

## Run

```bash
python run.py
```

If port **8000** is already in use (e.g. an old server still running), `run.py` picks the next free port (**8001**, **8002**, …) and prints the URL. Set `PORT=9000` to force a port.

Then open the printed URL (often `http://127.0.0.1:8000/`) — upload a **frontal face** image, choose a hairstyle, and call **POST `/try-on`**.

### API

- **POST** `/try-on` — multipart form: `image` (file), `hairstyle` (filename, e.g. `style_wavy_brown.png`)
- Returns **PNG** image bytes.

Results are saved under `outputs/results/`; logs under `outputs/logs/`.

## Model weights (auto-download)

On first startup, `app/backend/model_loader.py` downloads:

- `face_landmarker.task` — MediaPipe **Tasks** Face Landmarker (replaces legacy `mediapipe.solutions.face_mesh`, removed in MediaPipe 0.10+)
- `79999_iter.pth` — BiSeNet face parsing ([Hugging Face](https://huggingface.co/vivym/face-parsing-bisenet))
- `big-lama.pt` — LaMa inpainting ([Hugging Face `fashn-ai/LaMa`](https://huggingface.co/fashn-ai/LaMa))

Files are stored in `models/weights/`.

## LaMa FFC code

The `models/inpainting/lama/` package contains the **FFC ResNet generator** (Fourier convolutions) derived from the [LaMa](https://github.com/advimman/lama) project (MIT-style academic use; retain upstream license notices if you redistribute).

## Training scripts (stubs)

- `scripts/train_segmentation.py` — BiSeNet fine-tuning hook
- `scripts/train_inpainting.py` — LaMa-style generator training hook
- `scripts/train_gan.py` — small `HairRefinementNet` refinement

Wire these to your CSVs under `data/processed/` and `torch.utils.data.DataLoader` for full training loops.

## Project layout

See the repository tree: `app/backend` (API), `app/frontend` (UI), `models/` (BiSeNet, LaMa, refinement), `pipeline/` (steps + `full_pipeline.py`), `utils/`, `scripts/`, `data/`, `outputs/`.

## Troubleshooting

- **No face detected** — use a clear, frontal portrait; lighting should show the full face.
- **CUDA out of memory** — reduce input resolution in `models/inpainting/inference.py` (`DEFAULT_INPAINT_SIZE`) or segmentation input size in `models/segmentation/inference.py`.
- **First run is slow** — large checkpoint download for LaMa.
