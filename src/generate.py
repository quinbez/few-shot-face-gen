"""
generate.py — load the base model + trained LoRA and generate images.

- generate(...)          : make one or more images (returns a list of PIL images)
- load_pipeline(...)     : load base model + LoRA once (reused for batches)
- generate_for_eval(...) : make a batch across varied seeds for evaluation,
                           loading the model once and clearing GPU memory
                           between images to avoid out-of-memory errors.
"""
import gc
import torch
from diffusers import StableDiffusionPipeline
from src import config


def load_pipeline(lora_dir=None):
    """Load the base Stable Diffusion pipeline with our trained LoRA on top."""
    device = config.DEVICE
    lora_dir = lora_dir or config.LORA_DIR
    pipe = StableDiffusionPipeline.from_pretrained(config.BASE_MODEL)
    pipe.load_lora_weights(str(lora_dir))
    pipe = pipe.to(device)
    pipe.safety_checker = None            # avoid false-positive black images on faces
    return pipe


def generate(prompt=None, negative_prompt=None, n=1, steps=25,
             guidance_scale=7.5, seed=None, lora_dir=None,
             pipe=None, save_prefix="gen"):
    """Generate image(s) from a prompt. Returns a list of PIL images.

    Pass an already-loaded `pipe` to avoid reloading the model each call.
    Images are saved to outputs/generated/ without overwriting existing ones.
    """
    device = config.DEVICE
    prompt = prompt or config.INSTANCE_PROMPT
    if pipe is None:
        pipe = load_pipeline(lora_dir)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    images = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=n,
        generator=generator,
    ).images

    out_dir = config.LORA_DIR.parent / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    start = len(list(out_dir.glob(f"{save_prefix}_*.png")))
    for i, img in enumerate(images):
        img.save(out_dir / f"{save_prefix}_{start + i:03d}.png")
    print(f"Saved {len(images)} image(s) to {out_dir}")
    return images


def generate_for_eval(n=10, prompt=None, negative_prompt=None, steps=40,
                      guidance_scale=7.0, base_seed=0, lora_dir=None, clear_dir=True):
    """Generate a batch of n images across varied seeds, for evaluation.

    Loads the model once, generates one image per seed (base_seed .. base_seed+n-1),
    and clears GPU memory between images so it runs on a free Colab GPU.
    """
    out_dir = config.LORA_DIR.parent / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    if clear_dir:
        for p in out_dir.glob("gen_*.png"):
            p.unlink()

    pipe = load_pipeline(lora_dir)
    for k in range(n):
        generate(prompt=prompt, negative_prompt=negative_prompt, n=1, steps=steps,
                 guidance_scale=guidance_scale, seed=base_seed + k, pipe=pipe)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"Generated {n} images for evaluation in {out_dir}")
    return out_dir