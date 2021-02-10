import torch
import torchvision
from torch import nn
from settings import DEVICE

# Validation model
validation_model = torchvision.models.vgg16(pretrained=True)
validation_model.classifier = nn.Identity()
validation_model = validation_model.to(DEVICE)
validation_model.eval()


def l1_and_vgg_loss(sr: torch.tensor, hr: torch.tensor) -> torch.tensor:
    l1 = nn.L1Loss()(sr, hr)
    l1_features = nn.L1Loss()(validation_model(sr), validation_model(hr))
    return l1 + l1_features


def l1_loss(sr: torch.tensor, hr: torch.tensor) -> torch.tensor:
    l1 = nn.L1Loss()(sr, hr)
    return l1


def psnr(sr: torch.tensor, hr: torch.tensor) -> torch.tensor:
    mse = nn.MSELoss()(sr, hr)
    return 10 * torch.log10(1 / mse)

