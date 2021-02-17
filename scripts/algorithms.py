import torch
import torchvision
from torch import nn
from settings import DEVICE
import numpy as np


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class VGGPerceptual(nn.Module):
    def __init__(self, l1_weight=1, vgg_weight=1):
        super(VGGPerceptual, self).__init__()
        self.validation_model = torchvision.models.vgg16(pretrained=True).to(DEVICE)
        # Remove classifier part
        self.validation_model.classifier = nn.Identity()
        # Remove layers with deep features
        self.validation_model.features = nn.Sequential(*self.validation_model.features[:22])
        # Freeze model
        self.validation_model.eval()
        # Create L1 loss for measuring distance
        self.l1_loss = nn.L1Loss()
        # Set coeffs for losses
        self.l1_weight = l1_weight
        self.vgg_weight = vgg_weight

    def forward(self, input: torch.tensor, target: torch.tensor) -> torch.tensor:
        l1 = self.l1_loss(input, target)
        l1_features = self.l1_loss(self.validation_model(input), self.validation_model(target))
        return self.l1_weight * l1 + self.vgg_weight * l1_features


class PSNR:
    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(sr_img: torch.tensor, hr_img: torch.tensor) -> torch.tensor:
        with torch.no_grad():
            mse = torch.mean((torch.clamp(sr_img, min=-1, max=1) - hr_img) ** 2)
            return 10 * torch.log10(4.0 / mse)
