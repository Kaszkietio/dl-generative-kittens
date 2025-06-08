import torch
from torch import nn
import torchvision.transforms.functional as TF
import random


class AugmentPipeline(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if torch.rand(1).item() < self.p:
            # Pixel blitting: horizontal flip
            if torch.rand(1).item() < 0.5:
                x = TF.hflip(x)
            # Pixel blitting: 90-degree rotation
            if torch.rand(1).item() < 0.5:
                k = random.choice([0, 1, 2, 3])
                x = torch.rot90(x, k, [2, 3])
            # Pixel blitting: integer translation
            if torch.rand(1).item() < 0.5:
                max_shift = 4
                dx = random.randint(-max_shift, max_shift)
                dy = random.randint(-max_shift, max_shift)
                x = TF.affine(x, angle=0, translate=[dx, dy], scale=1.0, shear=[0.0, 0.0])

            # Geometric: random scaling & rotation
            angle = random.uniform(-15, 15)
            scale = random.uniform(0.9, 1.1)
            x = TF.affine(x, angle=angle, translate=[0, 0], scale=scale, shear=[0.0, 0.0])

            # Color: brightness, contrast, saturation
            x = TF.adjust_brightness(x, random.uniform(0.8, 1.2))
            x = TF.adjust_contrast(x, random.uniform(0.8, 1.2))
            x = TF.adjust_saturation(x, random.uniform(0.8, 1.2))

        return x