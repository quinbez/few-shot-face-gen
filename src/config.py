from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "data" / "train"
TEST_DIR = ROOT / "data" / "test"
LORA_DIR = ROOT / "outputs" / "lora"

BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"

KEYWORD = "sks"                 # rare token that will come to mean "him"
CLASS_WORD = "man"
INSTANCE_PROMPT = f"a photo of {KEYWORD} {CLASS_WORD}"

IMAGE_SIZE = 512
LORA_RANK = 4
LEARNING_RATE = 1e-4
TRAIN_STEPS = 5                 # tiny, just to test the loop on CPU
SEED = 42

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = get_device()