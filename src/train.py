import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from src import config, model, data
from diffusers import StableDiffusionPipeline


class _ClassDataset(Dataset):
    """Prior-preservation class images, paired with the class prompt."""
    def __init__(self):
        class_dir = Path(getattr(config, "CLASS_DIR", config.TRAIN_DIR.parent / "class"))
        self.paths = sorted(class_dir.glob("*.png")) + sorted(class_dir.glob("*.jpg"))
        if not self.paths:
            raise FileNotFoundError(f"No class images in {class_dir}. Run the class-image cell first.")
        self.prompt = getattr(config, "CLASS_PROMPT", f"a photo of a {config.CLASS_WORD}")
        self.tfm = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.CenterCrop(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return {"pixel_values": self.tfm(Image.open(self.paths[i]).convert("RGB")),
                "prompt": self.prompt}


def train(steps=None, lr=None):
    steps = steps or config.TRAIN_STEPS
    lr = lr or config.LEARNING_RATE
    device = config.DEVICE
    use_prior = getattr(config, "PRIOR_PRESERVATION", False)
    prior_w = getattr(config, "PRIOR_LOSS_WEIGHT", 1.0)

    # 1. Load models and add LoRA
    m = model.load_models(device=device)
    unet = model.add_lora(m["unet"])
    vae, text_encoder, tokenizer = m["vae"], m["text_encoder"], m["tokenizer"]
    noise_scheduler = m["noise_scheduler"]
    unet.train()

    # 2. Optimizer — only the LoRA params
    lora_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=lr)

    # 3. Data (instance + optional class images)
    loader = data.get_dataloader(batch_size=1)
    class_loader = class_iter = None
    if use_prior:
        class_loader = DataLoader(_ClassDataset(), batch_size=1, shuffle=True)
        class_iter = iter(class_loader)
        print(f"Prior preservation ON (weight={prior_w}).")

    # helper: one denoising MSE loss for a batch of images + prompts
    def denoise_loss(images, prompts):
        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample() * 0.18215
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                  (latents.shape[0],), device=device).long()
        noisy = noise_scheduler.add_noise(latents, noise, timesteps)
        tokens = tokenizer(prompts, padding="max_length",
                           max_length=tokenizer.model_max_length,
                           return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            enc = text_encoder(tokens)[0]
        pred = unet(noisy, timesteps, enc).sample
        return F.mse_loss(pred, noise)

    # 4. Training loop
    step = 0
    losses = []
    progress = tqdm(total=steps, desc="Training")
    while step < steps:
        for batch in loader:
            images = batch["pixel_values"].to(device)
            loss = denoise_loss(images, batch["prompt"])

            if use_prior:
                try:
                    cbatch = next(class_iter)
                except StopIteration:
                    class_iter = iter(class_loader)   # loop the class set
                    cbatch = next(class_iter)
                closs = denoise_loss(cbatch["pixel_values"].to(device), cbatch["prompt"])
                loss = loss + prior_w * closs

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            progress.update(1)
            progress.set_postfix(loss=f"{loss.item():.4f}")
            losses.append(loss.item())
            if step >= steps:
                break

    progress.close()
    config.LORA_DIR.mkdir(parents=True, exist_ok=True)
    from peft.utils import get_peft_model_state_dict
    lora_state = get_peft_model_state_dict(unet)
    StableDiffusionPipeline.save_lora_weights(config.LORA_DIR, unet_lora_layers=lora_state)
    print(f"Saved LoRA weights to {config.LORA_DIR}")
    return unet, losses