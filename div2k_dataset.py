import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import cv2
from skimage import io
import os


class DIV2KDataset(Dataset):
    def __init__(self, hr_folder: str, lr_folder: str, scale: int, transform=None):
        self.size = len(os.listdir(hr_folder))
        self.hr_folder = hr_folder
        self.lr_folder = lr_folder
        self.transform = transform
        self.scale = scale

    def __getitem__(self, item: int) -> (torch.tensor, torch.tensor):
        hr_image = torch.from_numpy(io.imread(self.hr_folder + f"/{item + 1:04}.png"))
        lr_image = torch.from_numpy(io.imread(self.lr_folder + f"/{item + 1:04}x{self.scale}.png"))

        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)

        return hr_image, lr_image

    def __len__(self) -> int:
        return self.size
