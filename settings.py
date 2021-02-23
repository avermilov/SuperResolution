import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINTS_PATH = "Checkpoints/"
LEARNING_RATE_NAME = "SR Learning Rate"
VALIDATION_NAME = "SR Validation Metric"
GENERATOR_LOSS_NAME = "SR Generator Loss"
DISCRIMINATOR_LOSS_NAME = "SR Discriminator Loss"
SUPERVISED_LOSS_NAME = "SR Supervised Loss"
