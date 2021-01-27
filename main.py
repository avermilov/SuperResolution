from skimage import io

from div2k_dataset import DIV2KDataset
from torchvision import transforms
import cv2

if __name__ == "__main__":
    transforms = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])
    ds = DIV2KDataset(hr_folder="DIV2K_train_HR/", lr_folder="DIV2K_train_LR_bicubic/X2", scale=2,
                      transform=transforms)


