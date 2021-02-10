import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from settings import *


def predict(net: nn.Module,
            epoch: int,
            predict_loader: DataLoader,
            lr_transform: nn.Module,
            sw: SummaryWriter = None) -> None:
    net.eval()
    with torch.no_grad():
        for i, hr_img in enumerate(predict_loader):
            hr_img = hr_img[0].to(DEVICE)
            lr_img = torch.unsqueeze(lr_transform(hr_img[0]), 0)
            res = net(lr_img)[0]
            res = torch.clamp(res / 2 + 0.5, 0, 1)
            if sw:
                sw.add_image(f"img{i:02}_pred", res, global_step=epoch)
            torchvision.utils.save_image(res, f"{PREDICT_TO_PATH}/img{i:02}_epoch{epoch:02}.jpg")
    net.train()
