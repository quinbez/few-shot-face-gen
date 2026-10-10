import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig
from src import config

def load_models(device=None, dtype=torch.float32):
    device = device or config.DEVICE
    name = config.BASE_MODEL

    tokenizer      = CLIPTokenizer.from_pretrained(name, subfolder="tokenizer")
    text_encoder   = CLIPTextModel.from_pretrained(name, subfolder="text_encoder")
    vae            = AutoencoderKL.from_pretrained(name, subfolder="vae")
    unet           = UNet2DConditionModel.from_pretrained(name, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(name, subfolder="scheduler")

    # freeze all base weights — only the LoRA layers (added next) will train
    for m in (vae, text_encoder, unet):
        m.requires_grad_(False)
        m.to(device, dtype=dtype)

    return {
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "vae": vae,
        "unet": unet,
        "noise_scheduler": noise_scheduler,
    }

def add_lora(unet, rank=None):
    """Attach trainable LoRA adapters to the U-Net's attention layers.
    Only these adapters train; the base weights stay frozen."""
    rank = rank or config.LORA_RANK

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_config)

    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in unet.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.3f}%)")

    return unet