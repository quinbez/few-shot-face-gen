import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from src import config


class FaceDataset(Dataset):
    """Loads training images and pairs each with the instance prompt."""

    def __init__(self, image_dir=None, size=None):
        image_dir = Path(image_dir or config.TRAIN_DIR)
        size = size or config.IMAGE_SIZE
        self.paths = sorted(image_dir.glob("*.png"))
        if not self.paths:
            raise FileNotFoundError(f"No .png images in {image_dir}")

        self.prompt = config.INSTANCE_PROMPT
        tfms = [
            transforms.Resize(config.IMAGE_SIZE),
            transforms.CenterCrop(config.IMAGE_SIZE),
        ]
        if getattr(config, "AUGMENT", False):
            tfms += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.0),
            ]
        tfms += [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
        self.transform = transforms.Compose(tfms)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return {"pixel_values": self.transform(img), "prompt": self.prompt}


def get_dataloader(batch_size=1, shuffle=True):
    dataset = FaceDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    print(f"Loaded {len(dataset)} training images from {config.TRAIN_DIR}")
    return loader