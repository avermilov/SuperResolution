import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from validate_net import validate
from predict import predict
from settings import *


def train_net(net: nn.Module,
              epochs: int,
              train_criterion: nn.Module,
              valid_criterion: nn.Module,
              optimizer: optim.Optimizer,
              train_loader: DataLoader,
              validation_loader: DataLoader,
              predict_loader: DataLoader,
              lr_transform: nn.Module,
              predict_transform: nn.Module,
              scheduler=None):
    net.train()
    sw = SummaryWriter()

    running_loss = 0
    total = 0

    minibatch_no = 0
    for epoch in range(epochs):
        if scheduler is not None:
            for g in optimizer.param_groups:
                g["lr"] = scheduler[epoch]

        for i, hr_img in tqdm(enumerate(train_loader)):
            lr_img = lr_transform(hr_img[0])
            hr_img, lr_img = hr_img[0].to(DEVICE), lr_img.to(DEVICE)
            total += lr_img.shape[0]

            sr_img = net(lr_img)
            loss = train_criterion(sr_img, hr_img)
            running_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % EVERY_N_MINIBATCHES == EVERY_N_MINIBATCHES - 1:
                sw.add_scalar(TRAIN_LOG_NAME, running_loss / total, minibatch_no)
                minibatch_no += 1

        with torch.no_grad():
            acc = validate(net, epoch, valid_criterion, validation_loader, lr_transform, sw)
            torch.save(net.state_dict(), CHECKPOINTS_PATH + f"epoch_{epoch:02}_acc_{acc:.3}.pth")
            predict(net, epoch, predict_loader, predict_transform, sw)

    sw.close()
