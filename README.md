# Few-Shot Face Generation

Fine-tuning a pretrained text-to-image model to generate new, realistic
images of one specific person from only a small, fixed set of photos.

## Overview

A normal image generator has never seen a specific private person, and a
photo editor can only change pictures that already exist. This project
teaches a pretrained diffusion model one person's face from ~15 photos,
then generates new images of that person in new poses and settings.

It is a **few-shot personalisation** task: learn an identity from a handful
of examples, then create new, identity consistent images on demand.

## Method

- **Base model:** Stable Diffusion v1.5 (latent diffusion), via Hugging Face
  `diffusers` in PyTorch.
- **Personalisation:** DreamBooth — the subject is bound to a rare keyword,
  `sks` (prompt: "a photo of sks man").
- **Efficient fine-tuning:** LoRA (Low-Rank Adaptation) on the U-Net
  attention layers — only ~0.1–0.4% of the model trains, so it runs on a
  free GPU. Two ranks were compared (4 and 16).
- **Generation:** classifier-free guidance with a negative prompt to remove
  the grainy/vintage look inherited from older training photos.

## Evaluation

Generated images are scored with **ArcFace** face embeddings (insightface):

- **Identity** — cosine similarity to 4 held out real photos (higher = more
  like the subject).
- **Face-detection rate** — fraction of outputs with a usable face.
- **Novelty** — similarity to the nearest training photo, as a check against
  copying.

Scores are calibrated against a no-LoRA baseline (~0.01, random) and a
real-photo ceiling (~0.47), with ~0.35 as the "same person" threshold.

## Workflow

Training runs on a free Google Colab GPU; the small LoRA adapter (a few MB)
is downloaded, and generation + evaluation run locally on a MacBook (Apple
M4). Heavy training in the cloud; fast, repeated generation and scoring
locally.

## Project structure

```text
few-shot-face-gen/
├── README.md
├── src/
│   ├── config.py       # paths, model name, prompt, LoRA rank, steps, threshold
│   ├── preprocess.py   # crop raw photos to 512x512 squares
│   ├── data.py         # dataset + dataloader (pairs each image with the prompt)
│   ├── model.py        # load frozen SD components + attach LoRA adapters
│   ├── train.py        # LoRA fine-tuning loop; saves the adapter
│   ├── generate.py     # load base model + LoRA, generate images
│   └── evaluate.py     # ArcFace identity, face-detection and novelty metrics
├── notebooks/
│   ├── train.ipynb     # run training
│   ├── generate.ipynb  # generate from the trained LoRA
│   └── evaluate.ipynb  # generate a batch + score it
├── data/               # (gitignored - private photos, not shared)
└── outputs/            # (gitignored - models and generated images)
```

## How to run

```bash
# 1. install
pip install torch diffusers transformers peft accelerate pillow insightface onnxruntime

# 2. preprocess raw photos -> data/train/ (512x512)
python src/preprocess.py

# 3. train (on a GPU; rank/steps set in src/config.py or overridden in the notebook)
#    produces outputs/lora/pytorch_lora_weights.safetensors

# 4. generate + evaluate (notebooks/generate.ipynb, notebooks/evaluate.ipynb)
```

## Note on data

The training photos are personal images and are **not** included in this
repository (`data/` and `outputs/` are gitignored). Only the code is shared.