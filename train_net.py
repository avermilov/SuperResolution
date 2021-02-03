import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from settings import *
from utils import *
from validate_net import validate

validation_model = torchvision.models.vgg16(pretrained=True)
validation_model.classifier = nn.Identity()
validation_model = validation_model.to(DEVICE)
validation_model.eval()


def train_net(net: nn.Module,
              epochs: int,
              train_criterion: nn.Module,
              valid_criterion: nn.Module,
              optimizer: optim.Optimizer,
              train_loader: DataLoader,
              validation_loader: DataLoader):
    net.train()
    sw = SummaryWriter()

    running_loss = 0
    total = 0

    for epoch in range(epochs):
        minibatch_no = 0
        for i, (lr_img, hr_img) in tqdm(enumerate(train_loader)):
            lr_img, hr_img = lr_img.to(DEVICE), hr_img.to(DEVICE)
            total += lr_img.shape[0]

            loss = train_criterion(lr_img, hr_img, net, validation_model)
            running_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % EVERY_N_MINIBATCHES == EVERY_N_MINIBATCHES - 1:
                sw.add_scalar(TRAIN_NAME, running_loss / total, minibatch_no)
                minibatch_no += 1

        with torch.no_grad():
            acc = validate(net, epoch, valid_criterion, validation_loader, sw)
            sw.add_scalar(VALID_NAME, acc, epoch)

    sw.close()
