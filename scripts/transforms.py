import os
from random import choice
from typing import List

import PIL
import scipy.io as sio
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Lists of kernels for training and validation.
train_kernels = None
valid_kernels = None

# Percentage of kernels and noises going to training pool.
TRAIN_PERCENTAGE = 0.8

# Lists of noises for training and validation.
train_noises = None
valid_noises = None

# Transforms for cropping noise for training and validation.
train_crop_transform = None
valid_crop_transform = None

train_bicubic = None
valid_bicubic = None


def load_kernels(kernels_path: str):
    global train_kernels, valid_kernels

    if kernels_path.lower() == "none":
        return

    kernels = []
    for filename in os.listdir(kernels_path):
        mat = sio.loadmat(os.path.join(kernels_path, filename))["Kernel"]
        mat = torch.from_numpy(mat)
        mat.requires_grad = False
        mat = torch.unsqueeze(mat, dim=0)
        mat = torch.unsqueeze(mat, dim=0)
        kernels.append(mat)

    KERNEL_TRAIN_SIZE = int(TRAIN_PERCENTAGE * len(kernels))
    train_kernels, valid_kernels = torch.utils.data.random_split(
        kernels, [KERNEL_TRAIN_SIZE, len(kernels) - KERNEL_TRAIN_SIZE]
    )

    train_kernels = [kernel for kernel in train_kernels]
    valid_kernels = [kernel for kernel in valid_kernels]


def load_noises(noises_path: str) -> None:
    global train_noises, valid_noises

    if noises_path == "none":
        return

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

    if train_kernels is not None:
        images = apply_kernel(images, train_kernels)
    else:
        images = train_bicubic(images)

    if train_noises is not None:
        images = inject_noise(images, train_noises, train_crop_transform)

    images = torch.clamp((images - 0.5) / 0.5, min=-1, max=1)

    return images


def get_train_lr_transform(scale: int, hr_crop: int):
    global train_crop_transform, train_bicubic
    train_crop_transform = transforms.RandomCrop(hr_crop // scale)
    train_bicubic = transforms.Resize(hr_crop // scale, interpolation=PIL.Image.BICUBIC)
    return lambda x: train_lr_transform(x)


def validation_lr_transform(images: torch.tensor):
    images = torch.clamp((images + 1) / 2, min=0, max=1)

    if valid_kernels is not None:
        images = apply_kernel(images, valid_kernels)
    else:
        images = valid_bicubic(images)

    if valid_noises is not None:
        images = inject_noise(images, valid_noises, valid_crop_transform)

    images = torch.clamp((images - 0.5) / 0.5, min=-1, max=1)

    return images


def get_validation_lr_transform(scale: int, hr_crop: int):
    global valid_crop_transform, valid_bicubic
    valid_crop_transform = transforms.RandomCrop(hr_crop // scale)
    valid_bicubic = transforms.Resize(hr_crop // scale, interpolation=PIL.Image.BICUBIC)
    return lambda x: validation_lr_transform(x)
