import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_PATH = "DIV2K_train_HR/"
VALIDATION_PATH = "DIV2K_valid_HR/"
CHECKPOINTS_PATH = "Checkpoints/"
LEARNING_RATE_NAME = "SR Learning Rate"
TRAIN_NAME = "SR Training"
VALIDATION_NAME = "SR Validation"

EVERY_N_MINIBATCHES = 10
MAX_IMAGES_LOG = 10
