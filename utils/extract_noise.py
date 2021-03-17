import argparse

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm


def extract(from_path: str, extract_to_path: str, denoise_strength: int, window_size: int) -> None:
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    ds = ImageFolder(from_path, transform=transform)
    loader = torch.utils.data.DataLoader(ds, batch_size=1,
                                         shuffle=False, num_workers=0)

    i = 0
    with torch.no_grad():
        for (data, _) in tqdm(loader):
            source = data
            noisy = (np.transpose(source.squeeze(0).cpu().numpy(), (1, 2, 0)) * 255).astype(np.uint8)
            denoised = cv2.fastNlMeansDenoisingColored(
                noisy, None, denoise_strength, denoise_strength, window_size, window_size * 3
            )
            extracted_noise = noisy - denoised
            cv2.imwrite(extract_to_path + f'{i:06}__denoise{denoise_strength:02}_win{window_size:03}_noise.png',
                        cv2.cvtColor(extracted_noise + 128, cv2.COLOR_RGB2BGR))
            i += 1


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("from_path", type=str, default=None)
    argparser.add_argument("to_path", type=str, default=None)
    argparser.add_argument("noise_level", type=int, default=None)
    argparser.add_argument("window_size", type=int, default=None)
    args = argparser.parse_args()

    extract(args.from_path, args.to_path, args.noise_level, args.window_size)
