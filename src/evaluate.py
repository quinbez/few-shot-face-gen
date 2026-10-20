"""
evaluate.py — measure how well the generated images match the real person.

Uses ArcFace face embeddings (via the insightface library) to score identity
similarity between generated images and the held-out real test photos.

Metrics reported:
  - Identity similarity: cosine similarity between each generated face and the
    real test faces (0 = different person, 1 = identical). ArcFace convention:
    roughly > 0.35 indicates the same person.
  - Face-detection rate: fraction of generated images that contain a usable
    face at all (distorted outputs have none - a useful quality signal).
  - Novelty check: similarity of each generated face to the nearest TRAINING
    face, to confirm outputs are new images rather than copies of the
    training set (values well below 1.0 = novel, not memorised).

Install (Colab GPU):
    !pip install -q insightface onnxruntime-gpu
Install (local CPU / Mac):
    pip install insightface onnxruntime
"""
from pathlib import Path
import numpy as np
from PIL import Image

from src import config

_app = None
_IMG_EXTS = (".png", ".jpg", ".jpeg")


def _get_app():
    """Load the ArcFace model once (RetinaFace detector + ArcFace recogniser)."""
    global _app
    if _app is None:
        from insightface.app import FaceAnalysis
        # tries GPU first, falls back to CPU automatically
        _app = FaceAnalysis(name="buffalo_l",
                            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        _app.prepare(ctx_id=0, det_size=(640, 640))
    return _app


def get_embedding(image):
    """Return a normalised ArcFace embedding for the largest face, or None."""
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    rgb = np.array(image)
    bgr = rgb[:, :, ::-1]                       # insightface expects BGR
    faces = _get_app().get(bgr)
    if not faces:
        return None
    # if several faces are found, keep the largest by bounding-box area
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return faces[-1].normed_embedding


def _embeddings_for_dir(folder):
    folder = Path(folder)
    embs = {}
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in _IMG_EXTS:
            e = get_embedding(p)
            if e is not None:
                embs[p.name] = e
    return embs


def evaluate(generated_dir=None, test_dir=None, train_dir=None, threshold=None):
    """Score generated images against the real test photos.

    Returns a dict of summary metrics and prints a readable report.
    """
    generated_dir = Path(generated_dir or (config.LORA_DIR.parent / "generated"))
    test_dir = Path(test_dir or config.TEST_DIR)
    threshold = threshold if threshold is not None else getattr(config, "FACE_SIM_THRESHOLD", 0.35)

    # 1. reference embeddings from the real held-out photos
    ref = _embeddings_for_dir(test_dir)
    if not ref:
        raise RuntimeError(f"No faces detected in test images at {test_dir}")
    ref_mat = np.stack(list(ref.values()))                 # [R, 512]

    # 2. embeddings from the generated images
    gen = _embeddings_for_dir(generated_dir)
    n_total = len([p for p in generated_dir.iterdir() if p.suffix.lower() in _IMG_EXTS])
    n_faces = len(gen)

    # 3. optional training embeddings for the novelty (anti-copy) check
    train_mat = None
    if train_dir is not None:
        tr = _embeddings_for_dir(train_dir)
        if tr:
            train_mat = np.stack(list(tr.values()))

    print(f"Generated images:         {n_total}")
    print(f"  with a detectable face: {n_faces}  ({100 * n_faces / max(n_total, 1):.0f}%)")
    print(f"Real reference faces:     {len(ref)}")
    print("-" * 56)

    sims, novelty = [], []
    for name, e in gen.items():
        id_sim = float(np.mean(ref_mat @ e))               # mean cosine to real faces
        sims.append(id_sim)
        line = f"{name:>18}:  identity={id_sim:.3f}"
        if train_mat is not None:
            nn = float(np.max(train_mat @ e))              # nearest training face
            novelty.append(nn)
            line += f"   nearest-train={nn:.3f}"
        print(line)

    sims = np.array(sims)
    print("-" * 56)
    result = {"n_total": n_total, "n_faces": n_faces}
    if len(sims):
        passed = int((sims >= threshold).sum())
        print(f"Mean identity similarity: {sims.mean():.3f}")
        print(f"Best / worst:             {sims.max():.3f} / {sims.min():.3f}")
        print(f"Above threshold {threshold}:     {passed}/{len(sims)} "
              f"({100 * passed / len(sims):.0f}%)")
        if novelty:
            print(f"Mean nearest-train sim:   {np.mean(novelty):.3f} "
                  f"(well below 1.0 = novel, not copied)")
        result.update({
            "mean_identity": float(sims.mean()),
            "best_identity": float(sims.max()),
            "worst_identity": float(sims.min()),
            "pass_rate": float((sims >= threshold).mean()),
            "threshold": threshold,
            "similarities": sims.tolist(),
            "mean_novelty": float(np.mean(novelty)) if novelty else None,
        })
    return result


if __name__ == "__main__":
    evaluate(train_dir=config.TRAIN_DIR)