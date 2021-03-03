import torch
import torchvision
from torch import nn as nn

from settings import DEVICE


class LSGANDisLoss(nn.Module):
    def __init__(self):
        super(LSGANDisLoss, self).__init__()

    def forward(self, fake: torch.tensor, real: torch.tensor) -> torch.tensor:
        return torch.mean(fake ** 2 + (real - 1) ** 2)


class LSGANGenLoss(nn.Module):
    def __init__(self):
        super(LSGANGenLoss, self).__init__()

    def forward(self, fake: torch.tensor, real: torch.tensor) -> torch.tensor:
        return torch.mean((fake - 1) ** 2)


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
        self.l1_loss = nn.L1Loss()

    def forward(self, input: torch.tensor, target: torch.tensor) -> torch.tensor:
        l1 = self.l1_loss(input, target)
        l1_features = self.l1_loss(self.validation_model(input), self.validation_model(target))
        return self.l1_coeff * l1 + self.vgg_coeff * l1_features
