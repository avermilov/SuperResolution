import cv2
import torch
import scipy.io as sio
import torch.nn.functional as F
import torch.nn as nn
from random import choice

from torchvision import transforms

kernel_paths = ["kernels/frame812_kernel_x2.mat",
                "kernels/frame839_kernel_x2.mat"]

kernels = [sio.loadmat(path)["Kernel"] for path in kernel_paths]


# todo: use conv2d?
def train_lr_transform(images: torch.tensor, lr_scale: int):
    images = torch.clamp((images + 1) / 2, min=0, max=1)
    images = images

    for i in range(images.shape[0]):
        img = images[i]
        img = img.numpy().transpose(1, 2, 0)
        ker = choice(kernels)
        img = cv2.filter2D(img, -1, ker)
        images[i] = torch.from_numpy(img).permute(2, 0, 1)

    images = torch.clamp((images - 0.5) / 0.5, min=-1, max=1)
    images = transforms.Resize((lr_scale, lr_scale))(images)
    return images


def get_train_lr_transform(scale: int, hr_crop: int):
    return lambda x: train_lr_transform(x, hr_crop // scale)


def validation_lr_transform(images: torch.tensor, lr_scale: int):
    images = torch.clamp((images + 1) / 2, min=0, max=1)
    images = images

    for i in range(images.shape[0]):
        img = images[i]
        img = img.numpy().transpose(1, 2, 0)
        ker = choice(kernels)
        img = cv2.filter2D(img, -1, ker)
        images[i] = torch.from_numpy(img).permute(2, 0, 1)

    images = torch.clamp((images - 0.5) / 0.5, min=-1, max=1)
    images = transforms.Resize((lr_scale, lr_scale))(images)
    return images


def get_validation_lr_transform(scale: int, hr_crop: int):
    return lambda x: validation_lr_transform(x, hr_crop // scale)

