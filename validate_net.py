from torch import nn
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from settings import DEVICE, NUM_IMAGES
from utils import create_grid


def validate(net: nn.Module, epoch: int, criterion: nn.Module, validation_loader: DataLoader, sw: SummaryWriter) -> float:
    net.eval()

    acc = 0
    logged = False
    with torch.no_grad():
        for i, (lr_img, hr_img) in enumerate(validation_loader):
            lr_img, hr_img = lr_img.to(DEVICE), hr_img.to(DEVICE)
            if not logged:
                results = net(lr_img).to(DEVICE)
                for j in range(min(NUM_IMAGES, lr_img.shape[0])):
                    sw.add_image(f"img{j + 1}",
                                 create_grid(lr_img[j].cpu(), results[j].cpu(), hr_img[j].cpu(), normalized=True),
                                 epoch)
                logged = True
            lr_img, hr_img = lr_img.to(DEVICE), hr_img.to(DEVICE)

            acc += criterion(lr_img, hr_img, net).item()

    return acc / len(validation_loader)
