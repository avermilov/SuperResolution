import cv2
import torch
import os
import scipy.io as sio
import torch.nn.functional as F
import torch.nn as nn
from random import choice
import numpy as np
from PIL import Image

from torchvision import transforms
from typing import List

from torchvision.datasets import ImageFolder

KERNEL_PATH = "kernels/"
kernels = []
for filename in os.listdir(KERNEL_PATH):
    mat = sio.loadmat(os.path.join(KERNEL_PATH, filename))["Kernel"]
    mat = torch.from_numpy(mat)
    mat.requires_grad = False
    mat = torch.unsqueeze(mat, dim=0)
    mat = torch.unsqueeze(mat, dim=0)
    kernels.append(mat)
TRAIN_PERCENTAGE = 0.8
KERNEL_TRAIN_SIZE = int(TRAIN_PERCENTAGE * len(kernels))

train_kernels, valid_kernels = torch.utils.data.random_split(
    kernels, [KERNEL_TRAIN_SIZE, len(kernels) - KERNEL_TRAIN_SIZE]
)
train_kernels = [kernel for kernel in train_kernels]
valid_kernels = [kernel for kernel in valid_kernels]

noises_path = "noises/"
noises_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
noises_ds = ImageFolder(noises_path, transform=noises_transforms)
NOISES_TRAIN_SIZE = int(TRAIN_PERCENTAGE * len(noises_ds))
train_noises, valid_noises = torch.utils.data.random_split(
    noises_ds, [NOISES_TRAIN_SIZE, len(noises_ds) - NOISES_TRAIN_SIZE]
)
train_noises = [noise[0] for noise in train_noises]
valid_noises = [noise[0] for noise in valid_noises]

train_crop_transform = None
valid_crop_transform = None


# for kernel in train_kernels:
#     print(np.max(kernel))

def inject_noise(images: torch.tensor, noises_list: List[torch.tensor], crop_transform) -> torch.tensor:
    noise = choice(noises_list)
    noise = crop_transform(noise)
    images += noise
    images = torch.clamp(images, -1, 1)
    return images


def apply_kernel(images: torch.tensor, kernels_list: List[torch.tensor]):
    kernel = choice(kernels_list)
    padding = (kernel.shape[-1] - 1) // 2
    images = F.pad(images, [padding] * 4, mode="reflect")
    downscaled = torch.cat([
        F.conv2d(images[:, i:i + 1, :, :], kernel, stride=2)
        for i in range(images.shape[1])],
        dim=1
    )

    return downscaled


def train_lr_transform(images: torch.tensor):
    images = torch.clamp((images + 1) / 2, min=0, max=1)

    images = apply_kernel(images, train_kernels)

    images = inject_noise(images, train_noises, train_crop_transform)

    images = torch.clamp((images - 0.5) / 0.5, min=-1, max=1)

    return images


def get_train_lr_transform(scale: int, hr_crop: int):
    global train_crop_transform
    train_crop_transform = transforms.RandomCrop(hr_crop // scale)
    return lambda x: train_lr_transform(x)


def validation_lr_transform(images: torch.tensor):
    images = torch.clamp((images + 1) / 2, min=0, max=1)

    images = apply_kernel(images, valid_kernels)

    images = inject_noise(images, valid_noises, valid_crop_transform)

    images = torch.clamp((images - 0.5) / 0.5, min=-1, max=1)

    return images


def get_validation_lr_transform(scale: int, hr_crop: int):
    global valid_crop_transform
    valid_crop_transform = transforms.RandomCrop(hr_crop // scale)
    return lambda x: validation_lr_transform(x)
