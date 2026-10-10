import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from src import config, model, data
from diffusers import StableDiffusionPipeline

def train(steps=None, lr=None):
    steps = steps or config.TRAIN_STEPS
    lr = lr or config.LEARNING_RATE
    device = config.DEVICE

    # 1. Load models and add LoRA
    m = model.load_models(device=device)
    unet = model.add_lora(m["unet"])
    vae, text_encoder, tokenizer = m["vae"], m["text_encoder"], m["tokenizer"]
    noise_scheduler = m["noise_scheduler"]
    unet.train()

    # 2. Optimizer — only the LoRA params (the ones that require grad)
    lora_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=lr)

    # 3. Data
    loader = data.get_dataloader(batch_size=1)

    # 4. Training loop
    step = 0
    losses = []
    progress = tqdm(total=steps, desc="Training")
    while step < steps:
        for batch in loader:
            images = batch["pixel_values"].to(device)

            # encode image -> latent, scale as SD expects
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample() * 0.18215

            # sample random noise and timesteps
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=device
            ).long()
            noisy = noise_scheduler.add_noise(latents, noise, timesteps)

            # encode the text prompt
            tokens = tokenizer(
                batch["prompt"], padding="max_length",
                max_length=tokenizer.model_max_length, return_tensors="pt"
            ).input_ids.to(device)
            with torch.no_grad():
                encoder_hidden_states = text_encoder(tokens)[0]

            # U-Net predicts the noise; loss = how wrong it was
            pred = unet(noisy, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(pred, noise)

            # backprop + update LoRA weights
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