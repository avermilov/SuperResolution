import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from settings import DEVICE
import os


def inference(net: nn.Module, epoch: int, inference_loader: DataLoader, save_path: str) -> None:
    net.eval()
    count = 0
    with torch.no_grad():
        for (images, _) in inference_loader:
            images = images.to(DEVICE)
            sr_images = net(images)
            sr_images = (sr_images + 1) / 2
            sr_images = torch.clamp(sr_images, 0, 1)
            for i in range(images.shape[0]):
                sr_path = save_path + f"sr{count:05}/"
                if not os.path.exists(sr_path):
                    os.mkdir(sr_path)
                torchvision.utils.save_image(sr_images[i], sr_path + f"sr{count:05}_ep{epoch:03}.png")
                count += 1
    net.train()
