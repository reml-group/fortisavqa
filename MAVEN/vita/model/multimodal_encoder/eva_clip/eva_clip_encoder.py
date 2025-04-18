import torch
import torch.nn as nn

from .eva_clip_processors import EvaClipImageTrainProcessor
from .eva_vit import Eva2LargePlusEncoder


class EvaClipVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_path = vision_tower
        self.config = VisionTowerConfig()

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = self.config

    def load_model(self):
        self.image_processor = EvaClipImageTrainProcessor(self.config.image_size)
        self.vision_tower = Eva2LargePlusEncoder(self.vision_tower_path)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0)
                ).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_features = self.vision_tower(images.to(device=self.device, dtype=self.dtype)).to(
                images.dtype
            )

        return image_features

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


class VisionTowerConfig:
    def __init__(self):
        self.image_size = 336
        self.patch_size = 14
        self.hidden_size = 1024
