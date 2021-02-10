import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
# from utils import *

from models import RDN
from utils import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Path names
TRAIN_NAME = "SR_Train"
VALID_NAME = "SR_Valid"
IMG_NAME = "SR_img"
RESULTS_PATH = "results/"
PREDICT_FROM_PATH = "ValidationImages/"
PREDICT_TO_PATH = "ValidationResults/"

# Tensorboard logging
NUM_IMAGES = 5
EVERY_N_MINIBATCHES = 10

# Transforms
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomCrop(64, 64)])
screwed_transform = transforms.Compose([
    transforms.Resize((32, 32))
])

# Dataset and loaders
ds = torchvision.datasets.ImageFolder("DIV2K_HR/", transform=data_transform)
train_set, valid_set = torch.utils.data.random_split(ds, [800, 100])
train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=10)
valid_loader = DataLoader(valid_set, batch_size=4, shuffle=False, num_workers=10)
predict_ds = torchvision.datasets.ImageFolder(PREDICT_FROM_PATH, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
predict_loader = DataLoader(predict_ds, shuffle=False, batch_size=1)

# Trajn specifics
scheduler = [0.001] * 10 #+ [0.0002] * 5 + [0.0001] * 10
epochs = 10
net = RDN.RDN(2, 3, 64, 64, 16, 8).to(DEVICE)
opt = torch.optim.Adam(net.parameters())
train_crit = l1_and_vgg_loss
valid_crit = PSNR
