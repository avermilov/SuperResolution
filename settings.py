import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Log names
TRAIN_LOG_NAME = "SR_Train"
VALID_LOG_NAME = "SR_Valid"
IMG_NAME = "SR_img"

# Path names
DATASET_PATH = "DIV2K_HR/"
CHECKPOINTS_PATH = "results/"
PREDICT_FROM_PATH = "ValidationImages/"
PREDICT_TO_PATH = "ValidationResults/"

# Tensorboard logging
NUM_IMAGES = 5
EVERY_N_MINIBATCHES = 10
