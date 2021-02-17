from torch import nn, optim
from torch.utils.data import DataLoader
from typing import List

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from scripts.validation import validate

from settings import *


def train(net: nn.Module,
          epochs: int,
          train_criterion,
          validation_criterion,
          optimizer: optim.Optimizer,
          train_loader: DataLoader,
          validation_loader: DataLoader,
          lr_transform: transforms.Compose,
          validation_transform: transforms.Compose,
          scheduler: List[float] = None,
          warmup: List[float] = None,
          summary_writer: SummaryWriter = None,
          max_images=0):
    net.train()

    running_loss = 0
    total_images = 0
    minibatch_number = 0
    for epoch in range(epochs):
        # If scheduler was passed, change lr to the one specified at each epoch.
        if scheduler:
            for g in optimizer.param_groups:
                g["lr"] = scheduler[epoch]

        for i, hr_image in tqdm(enumerate(train_loader)):
            # If warmup was passed, apply to mini_batch only during the first epoch.
            if warmup:
                if i < len(warmup):
                    for g in optimizer.param_groups:
                        g["lr"] = warmup[i]
                else:
                    warmup = None

            # Get proper hr minibatch and lr minibatch
            hr_image = hr_image[0]
            lr_image = lr_transform(hr_image)
            lr_image, hr_image = lr_image.to(DEVICE), hr_image.to(DEVICE)

            # Update amount of images looped over, calculate loss and add to running.
            total_images += hr_image.shape[0]
            sr_image = net(lr_image)
            loss = train_criterion(sr_image, hr_image)
            running_loss += loss.item()

            # Do backprop
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Update learning rate and loss to logging
            if summary_writer:
                learning_rate = next(iter(optimizer.param_groups))["lr"]
                summary_writer.add_scalar(LEARNING_RATE_NAME, learning_rate,
                                          global_step=i + epoch * len(train_loader))

                if (i + 1) % EVERY_N_MINIBATCHES == 0:
                    summary_writer.add_scalar(TRAIN_NAME, running_loss / total_images, global_step=minibatch_number)
                    minibatch_number += 1

            # Validate model
        with torch.no_grad():
            acc = validate(net, validation_criterion, validation_loader,
                           validation_transform, epoch, summary_writer, max_images)
            torch.save(net.state_dict(), CHECKPOINTS_PATH + f"Epoch{epoch:03}_Acc{acc:.3}.pth")
