import torch
from diffusers import StableDiffusionPipeline
from src import config


def generate(prompt=None, n=1, steps=25, seed=None, lora_dir=None):
    device = config.DEVICE
    lora_dir = lora_dir or config.LORA_DIR
    prompt = prompt or config.INSTANCE_PROMPT

    # load the base pipeline, then our trained LoRA on top
    pipe = StableDiffusionPipeline.from_pretrained(config.BASE_MODEL)
    pipe.load_lora_weights(str(lora_dir))   # modern loader (matches PEFT save)
    pipe = pipe.to(device)
    pipe.safety_checker = None
    
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    images = pipe(
        prompt, num_inference_steps=steps,
        num_images_per_prompt=n, generator=generator,
    ).images

    config.LORA_DIR.parent.mkdir(parents=True, exist_ok=True)
    out_dir = config.LORA_DIR.parent / "generated"
    out_dir.mkdir(exist_ok=True)
    paths = []
    for i, img in enumerate(images):
        p = out_dir / f"gen_{i:02d}.png"
        img.save(p)
        paths.append(p)
    print(f"Saved {len(paths)} image(s) to {out_dir}")
    return images