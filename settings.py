import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINTS_PATH = "Checkpoints/"
LEARNING_RATE_NAME = "SR Learning Rate"
GENERATOR_LOSS_NAME = "SR Generator Loss"
DISCRIMINATOR_LOSS_NAME = "SR Discriminator Loss"
SUPERVISED_LOSS_NAME = "SR Supervised Loss"
METRIC_NAMES = {
    "psnr": "SR Metrics/SR PSNR",
    "lpips_alex": "SR Metrics/SR LPIPS ALEX",
    "lpips_vgg": "SR Metrics/SR LPIPS VGG",
    "ssim": "SR Metrics/SR SSIM"
}
