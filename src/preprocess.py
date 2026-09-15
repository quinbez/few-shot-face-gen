"""
preprocess.py — crop raw photos to 512x512 squares.

Reads from data/bzr_data/, run:
    python src/preprocess.py
Results land in data/train/.
"""

from pathlib import Path
from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener()


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp"}


def square_crop(img, size=512):
    """Center-crop to a square, then resize — so nothing gets stretched."""
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    return img.resize((size, size), Image.LANCZOS)


def process_folder(src, dst, size=512):
    """Crop + resize every image in `src`, saving numbered PNGs into `dst`."""
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        print(f"No images found in {src}")
        return []

    print(f"Found {len(files)} images in {src}")
    saved = []
    for i, path in enumerate(files, 1):
        img = Image.open(path).convert("RGB")
        out = dst / f"{i:02d}.png"
        square_crop(img, size).save(out)
        saved.append(out)
        print(f"  {path.name}  ->  {out.name}")

    print(f"Done. {len(saved)} images saved to {dst}")
    return saved


if __name__ == "__main__":
    process_folder("data/bzr_data", "data/train", size=512)