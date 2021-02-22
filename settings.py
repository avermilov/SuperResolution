import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINTS_PATH = "Checkpoints/"
LEARNING_RATE_NAME = "SR Learning Rate"
TRAIN_NAME = "SR Training"
VALIDATION_NAME = "SR Validation"

