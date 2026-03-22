"""
Download sample subsets and write metadata CSVs for training scripts.

1) CelebAMask-HQ: sample images from Hugging Face (if available) or creates placeholder CSV.
2) FFHQ: sample subset URLs (thumbnails) — optional small download.
3) Helen: optional tiny sample list.

Run: python scripts/download_datasets.py
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"


def ensure_dirs():
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROC.mkdir(parents=True, exist_ok=True)


def try_download(url: str, dest: Path, timeout: int = 60) -> bool:
    try:
        r = requests.get(url, stream=True, timeout=timeout)
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
        return True
    except Exception as e:
        LOGGER.warning("Download failed %s: %s", url, e)
        return False


def write_celebamask_csv(rows):
    p = DATA_PROC / "celebamask_sample.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "mask_path", "labels"])
        w.writerows(rows)
    LOGGER.info("Wrote %s", p)


def write_ffhq_csv(rows):
    p = DATA_PROC / "ffhq_sample.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "mask_path", "labels"])
        w.writerows(rows)
    LOGGER.info("Wrote %s", p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffhq-samples", type=int, default=3, help="Number of FFHQ sample images to try fetching.")
    args = parser.parse_args()
    ensure_dirs()

    # CelebAMask-HQ: public sample on HF (dataset may change; we record intent in CSV).
    celeba_rows = []
    sample_img = "https://huggingface.co/datasets/akhaliq/CelebA-faces/resolve/main/img00000000.jpg"
    dest = DATA_RAW / "celebamask" / "sample_00.jpg"
    if try_download(sample_img, dest):
        celeba_rows.append([str(dest.relative_to(ROOT)), "", "face_sample"])
    write_celebamask_csv(celeba_rows)

    # FFHQ: best-effort URLs (repository layout may change; rows may stay empty).
    ffhq_urls = [
        "https://github.com/NVlabs/ffhq-dataset/raw/master/images1024x1024/00000.png",
    ]
    ffhq_rows = []
    for i, url in enumerate(ffhq_urls[: args.ffhq_samples]):
        d = DATA_RAW / "ffhq" / f"sample_{i:03d}.png"
        if try_download(url, d):
            ffhq_rows.append([str(d.relative_to(ROOT)), "", "ffhq_face"])
    write_ffhq_csv(ffhq_rows)

    # Helen optional placeholder CSV
    helen_p = DATA_PROC / "helen_sample.csv"
    with open(helen_p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "mask_path", "labels"])
        w.writerow(["", "", "optional_helen_not_downloaded"])
    LOGGER.info("Wrote %s (placeholder)", helen_p)


if __name__ == "__main__":
    main()
