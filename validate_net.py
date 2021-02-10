import torchvision
from torch import nn
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from settings import *


def validate(net: nn.Module, epoch: int, criterion: nn.Module, validation_loader: DataLoader, lr_transform: nn.Module,
             sw: SummaryWriter = None) -> float:
    net.eval()

    running_psnr = 0
    with torch.no_grad():
        for i, hr_img in enumerate(validation_loader):
            lr_img = lr_transform(hr_img[0])
            hr_img, lr_img = hr_img[0].to(DEVICE), lr_img.to(DEVICE)

            sr_img = net(lr_img)
            loss = criterion(sr_img, hr_img)

            running_psnr += loss.item()

    psnr = running_psnr / len(validation_loader)
    if sw:
        sw.add_scalar(VALID_LOG_NAME, psnr, epoch)

    return psnr
