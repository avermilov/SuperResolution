import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVERY_N_MINIBATCHES = 10
TRAIN_NAME = "SR_Train"
VALID_NAME = "SR_Valid"
IMG_NAME = "SR_img"
NUM_IMAGES = 5
PATH = "results/"
