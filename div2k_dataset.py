import torchvision
from torch.utils.data import Dataset
import os
import torch
from torch.utils.data import DataLoader
from skimage import io
from torchvision import transforms
import cv2

from typing import Tuple


class DIV2KDataset(Dataset):
    def __init__(self,
                 hr_folder: str,
                 lr_folder: str,
                 dcrop: Tuple[int, int],
                 lr_transform,
                 aug_transform=None,
                 normalize=False):
        self.size = len(os.listdir(hr_folder))
        self.hr_folder = hr_folder
        self.lr_folder = lr_folder
        self.aug_transform = aug_transform
        self.lr_transform = lr_transform
        self.dcrop = dcrop
        self.normalize = None if not normalize else transforms.Normalize((128, 128, 128), (128, 128, 128))

    def __getitem__(self, item: int) -> (torch.tensor, torch.tensor):
        hr_image = torch.from_numpy(io.imread(self.hr_folder + f"/{item + 1:04}.png"))

        if self.aug_transform:
            hr_image = self.aug_transform(hr_image)

        crp = torchvision.transforms.RandomCrop(self.dcrop)

        hr_image = crp(torch.transpose(hr_image, 0, 2))

        # lr_image = torch.from_numpy(
        #     cv2.resize(torch.transpose(hr_image, 0, 2).numpy(),
        #                (self.dcrop[0] // self.scale, self.dcrop[1] // self.scale)))
        # lr_image = torch.transpose(lr_image, 0, 2)

        lr_image = self.lr_transform(hr_image)

        if self.normalize:
            lr_image = self.normalize(lr_image.type(torch.FloatTensor))
            hr_image = self.normalize(hr_image.type(torch.FloatTensor))

        return lr_image, hr_image

    def __len__(self) -> int:
        return self.size

# transforms = transforms.Compose(
#     [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])
# ds = DIV2KDataset(hr_folder="DIV2K_HR/", lr_folder="DIV2K_LR_bicubic/X2", scale=2, dcrop=(200, 200),
#                   transform=None)
# print(next(iter(ds)))

# for i, (hr_imgs, lr_imgs) in enumerate(train_loader):
#     print(hr_imgs.shape)
#     break
