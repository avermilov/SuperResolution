import os
from random import sample, choice
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

# Scale parameter.
SCALE = None


def load_kernels(kernels_path: str, scale: int) -> None:
    global train_kernels, valid_kernels, SCALE

    if kernels_path is None:
        return

    SCALE = scale

    kernels = []
    for filename in os.listdir(kernels_path):
        mat = sio.loadmat(os.path.join(kernels_path, filename))["Kernel"]
        mat = torch.from_numpy(mat)
        mat.requires_grad = False
        # mat = torch.stack([mat, mat, mat])
        mat = torch.unsqueeze(mat, dim=0)
        # mat = torch.unsqueeze(mat, dim=0)
        mat = mat.type(torch.FloatTensor)
        kernels.append(mat)

    KERNEL_TRAIN_SIZE = int(TRAIN_PERCENTAGE * len(kernels))
    train_kernels, valid_kernels = torch.utils.data.random_split(
        kernels, [KERNEL_TRAIN_SIZE, len(kernels) - KERNEL_TRAIN_SIZE]
    )

    train_kernels = [kernel for kernel in train_kernels]
    valid_kernels = [kernel for kernel in valid_kernels]


def load_noises(noises_path: str) -> None:
    global train_noises, valid_noises

    if noises_path is None:
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

    for train_noise in train_noises:
        train_noise.requires_grad = False
    for valid_noise in valid_noises:
        valid_noise.requires_grad = False


def inject_noise(images: torch.tensor, noises_list: List[torch.tensor], crop_transform) -> torch.tensor:
    mini_batch_size = images.shape[0]
    noises = sample(noises_list, mini_batch_size)
    noises = torch.stack(noises)
    noise = crop_transform(noises)
    images += noise
    images = torch.clamp(images, -1, 1)
    return images


def apply_single_kernel(image: torch.tensor, kernel: torch.tensor) -> torch.tensor:
    # apply kernel to every channel individually
    res = [F.conv2d(image[:, i:i + 1, :, :], kernel, stride=SCALE) for i in range(image.shape[1])]
    return torch.cat(res, dim=1)[0]


def apply_kernel(images: torch.tensor, kernels_list: List[torch.tensor]):
    # kernels has size [n, 3, h, w]
    kernels = sample(kernels_list, images.shape[0])
    kernels = torch.stack(kernels)

    # calculate padding
    padding = (kernels.shape[-1] - 1) // 2

    # apply padding to mini batch
    images = F.pad(images, [padding] * 4, mode="reflect")

    # apply ind-th kernel to ind-th image individually
    results = [apply_single_kernel(images[ind:ind + 1], kernels[ind:ind + 1]) for ind in range(images.shape[0])]
    # make tensor out of list of tensor images
    downscaled = torch.stack(results)

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
