import torch
import torchvision
import torch.nn as nn
import torch.tensor as Tensor
import numpy as np

from settings import DEVICE


class VGGPerceptual(nn.Module):
    def __init__(self, l1_coeff: float = 1.0, vgg_coeff: float = 1.0):
        super(VGGPerceptual, self).__init__()
        # Set params
        self.l1_coeff = l1_coeff
        self.vgg_coeff = vgg_coeff
        # Get pretrained VGG model
        self.validation_model = torchvision.models.vgg16(pretrained=True).to(DEVICE)
        # Remove classifier part
        self.validation_model.classifier = nn.Identity()
        # Remove layers with deep features
        self.validation_model.features = nn.Sequential(*self.validation_model.features[:22])
        # Freeze model
        self.validation_model.eval()
        for param in self.validation_model.parameters():
            param.requires_grad = False
        # Create L1 loss for measuring distance
        self.loss = nn.L1Loss()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        l1 = self.loss(input, target)
        l1_features = self.loss(self.validation_model(input), self.validation_model(target))
        return self.l1_coeff * l1 + self.vgg_coeff * l1_features


class LSGANGenLoss(nn.Module):
    def __init__(self):
        super(LSGANGenLoss, self).__init__()

    def forward(self, fake: Tensor, real: Tensor) -> Tensor:
        return torch.mean((fake - 1) ** 2)


class LSGANDisLoss(nn.Module):
    def __init__(self):
        super(LSGANDisLoss, self).__init__()

    def forward(self, fake: Tensor, real: Tensor) -> Tensor:
        return torch.mean(fake ** 2 + (real - 1) ** 2)


class PSNR:
    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(sr_img: torch.tensor, hr_img: torch.tensor) -> torch.tensor:
        with torch.no_grad():
            mse = torch.mean((torch.clamp(sr_img, min=-1, max=1) - hr_img) ** 2)
            return 10 * torch.log10(4.0 / mse)


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
