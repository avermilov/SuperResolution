import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINTS_PATH = "Checkpoints/"
LOSSES_PATH_NAME = "SR Losses/"
LEARNING_RATE_NAME = "SR Learning Rate"

GENERATOR_LOSS_NAME = LOSSES_PATH_NAME + "SR Generator Loss"
GENERATOR_TOTAL_LOSS_NAME = LOSSES_PATH_NAME + "SR Generator Total Loss"
DISCRIMINATOR_TOTAL_LOSS_NAME = LOSSES_PATH_NAME + "SR Discriminator Total Loss"
DISCRIMINATOR_FAKE_LOSS_NAME = LOSSES_PATH_NAME + "SR Discriminator Fake Loss"
DISCRIMINATOR_REAL_LOSS_NAME = LOSSES_PATH_NAME + "SR Discriminator Real Loss"
SUPERVISED_LOSS_NAME = LOSSES_PATH_NAME + "SR Generator Supervised Loss"

METRICS_PATH_NAME = "SR Metrics/"
METRIC_NAMES = {
    "psnr": METRICS_PATH_NAME + "SR PSNR",
    "lpips_alex": METRICS_PATH_NAME + "SR LPIPS ALEX",
    "lpips_vgg": METRICS_PATH_NAME + "SR LPIPS VGG",
    "ssim": METRICS_PATH_NAME + "SR SSIM"
}
