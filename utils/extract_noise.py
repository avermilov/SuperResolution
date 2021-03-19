import argparse
import json

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm


def extract(from_path: str, extract_to_path: str, denoise_strength: int, window_size: int, kernel_size: int) -> None:
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    ds = ImageFolder(from_path, transform=transform)
    loader = torch.utils.data.DataLoader(ds, batch_size=32,
                                         shuffle=False, num_workers=0)

    i = 0
    with torch.no_grad():
        for (batch, _) in tqdm(loader):
            for data in batch:
                source = data
                noisy = (np.transpose(source.squeeze(0).cpu().numpy(), (1, 2, 0)) * 255).astype(np.uint8)
                denoised = cv2.fastNlMeansDenoisingColored(
                    noisy, None, denoise_strength, denoise_strength, window_size, window_size * 3
                )
                extracted_noise = noisy.astype(np.float32) - denoised.astype(np.float32)
                if kernel_size > 0:
                    kernel = np.ones((kernel_size, kernel_size), np.float32) / kernel_size ** 2
                    extracted_noise -= cv2.filter2D(extracted_noise, -1, kernel)

                extracted_noise -= np.mean(extracted_noise)
                cv2.imwrite(extract_to_path + f'{i:06}_s{denoise_strength:02}w{window_size:02}k{kernel_size:02}.png',
                            cv2.cvtColor(np.round(extracted_noise + 128).astype(np.uint8), cv2.COLOR_RGB2BGR))
                i += 1


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--json", type=str, default=None)
    args = argparser.parse_args()

    with open(args.json) as file:
        data = json.load(file)
        from_path = data["from_path"]
        to_path = data["to_path"]
        noise_level = data["noise_level"]
        window_size = data["window_size"]
        kernel_size = data["kernel_size"]

    extract(from_path, to_path, noise_level, window_size, kernel_size)
