import torch
import numpy as np


class PSNR:
    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(sr_img: torch.tensor, hr_img: torch.tensor) -> torch.tensor:
        with torch.no_grad():
            mse = torch.mean((torch.clamp(sr_img, min=-1, max=1) - hr_img) ** 2)
            return 10 * torch.log10(4.0 / mse)


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
