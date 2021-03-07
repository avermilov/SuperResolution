from torch import nn, optim
from torch.utils.data import DataLoader
from typing import List
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from scripts.validation import validate
from progressbar import progressbar
from settings import *


# def train(net: nn.Module,
#           epochs: int,
#           train_criterion,
#           validation_criterion,
#           optimizer: optim.Optimizer,
#           train_loader: DataLoader,
#           validation_loader: DataLoader,
#           lr_transform: transforms.Compose,
#           validation_transform: transforms.Compose,
#           scheduler: List[float] = None,
#           warmup: List[float] = None,
#           summary_writer: SummaryWriter = None,
#           max_images=0,
#           every_n: int = 1):
#     net.train()
#
#     running_loss = 0
#     total_images = 0
#     minibatch_number = 0
#     for epoch in range(epochs):
#         # If scheduler was passed, change lr to the one specified at each epoch.
#         if scheduler:
#             for g in optimizer.param_groups:
#                 g["lr"] = scheduler[epoch]
#
#         for i, hr_image in tqdm(enumerate(train_loader)):
#             # If warmup was passed, apply to mini_batch only during the first epoch.
#             if warmup:
#                 if i < len(warmup):
#                     for g in optimizer.param_groups:
#                         g["lr"] = warmup[i]
#                 else:
#                     warmup = None
#
#             # Get proper hr minibatch and lr minibatch
#             hr_image = hr_image[0]
#             lr_image = lr_transform(hr_image)
#             lr_image, hr_image = lr_image.to(DEVICE), hr_image.to(DEVICE)
#
#             # Update amount of images looped over, calculate loss and add to running.
#             total_images += hr_image.shape[0]
#             sr_image = net(lr_image)
#             loss = train_criterion(sr_image, hr_image)
#             running_loss += loss.item()
#
#             # Do backprop
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
#
#             # Update learning rate and loss to logging
#             if summary_writer:
#                 learning_rate = next(iter(optimizer.param_groups))["lr"]
#                 summary_writer.add_scalar(LEARNING_RATE_NAME, learning_rate,
#                                           global_step=i + epoch * len(train_loader))
#
#                 if (i + 1) % every_n == 0:
#                     summary_writer.add_scalar(SUPERVISED_LOSS_NAME, running_loss / total_images,
#                                               global_step=minibatch_number)
#                     minibatch_number += 1
#
#             # Validate model
#         with torch.no_grad():
#             acc = validate(net, validation_criterion, validation_loader,
#                            validation_transform, epoch, summary_writer, max_images)
#             torch.save(net.state_dict(), CHECKPOINTS_PATH + f"Epoch{epoch:03}_Acc{acc:.3}.pth")


def train_gan(generator: nn.Module,
              discriminator: nn.Module,
              supervised_criterion,
              gen_criterion,
              dis_criterion,
              gen_optimizer: optim.Optimizer,
              dis_optimizer: optim.Optimizer,
              gan_loss_coeff: float,
              start_epoch: int,
              epochs: int,
              validation_metric,
              train_loader: DataLoader,
              validation_loader: DataLoader,
              lr_transform: transforms.Compose,
              validation_transform: transforms.Compose,
              gen_scheduler: List[float] = None,
              dis_scheduler: List[float] = None,
              gen_warmup: List[float] = None,
              dis_warmup: List[float] = None,
              summary_writer: SummaryWriter = None,
              max_images=0,
              every_n: int = 1,
              best_metric: float = -1,
              save_name: str = ""):
    minibatch_number = len(train_loader) * start_epoch // every_n

    for epoch in range(start_epoch, epochs):
        # If scheduler was passed, change lr to the one specified at each epoch.
        if gen_scheduler:
            for g in gen_optimizer.param_groups:
                g["lr"] = gen_scheduler[epoch]
        if dis_scheduler:
            for g in dis_optimizer.param_groups:
                g["lr"] = dis_scheduler[epoch]

        generator.train()
        discriminator.train()

        # Zero all losses
        running_gen_loss = 0
        running_dis_loss = 0
        running_super_loss = 0
        total_images = 0

        pbar = tqdm(total=len(train_loader.dataset))
        for i, hr_images in enumerate(train_loader):
            # If warmup was passed, apply to mini_batch ONLY during the first epoch.
            if gen_warmup:
                if i < len(gen_warmup):
                    for g in gen_optimizer.param_groups:
                        g["lr"] = gen_warmup[i]
                else:
                    gen_warmup = None
            if dis_warmup:
                if i < len(dis_warmup):
                    for g in dis_optimizer.param_groups:
                        g["lr"] = dis_warmup[i]
                else:
                    dis_warmup = None

            # Get proper hr minibatch and lr minibatch.
            hr_images = hr_images[0]
            lr_images = lr_transform(hr_images)
            lr_images, hr_images = lr_images.to(DEVICE), hr_images.to(DEVICE)

            # Update cumulatively iterated over images count.
            total_images += hr_images.shape[0]

            # Zero the optimizers
            gen_optimizer.zero_grad()
            dis_optimizer.zero_grad()

            # Get model output
            sr_images = generator(lr_images)

            # Get concatenated images for conditional GAN
            scaled_lr_images = F.interpolate(lr_images, scale_factor=2, mode="bicubic", align_corners=True)
            concat_outputs = torch.cat((sr_images, scaled_lr_images), 1).to(DEVICE)
            concat_hr = torch.cat((hr_images, scaled_lr_images), 1).to(DEVICE)

            discriminator.requires_grad(False)
            # Calculate supervised loss and total generator loss.
            supervised_loss = supervised_criterion(sr_images, hr_images)
            dis_out = discriminator(concat_outputs)
            dis_hr = discriminator(concat_hr)
            generator_loss = supervised_loss + \
                             gan_loss_coeff * gen_criterion(dis_out, dis_hr)
            # Update cumulative losses counts.
            running_super_loss += supervised_loss.item()
            running_gen_loss += generator_loss.item()

            discriminator.requires_grad(True)
            concat_outputs = concat_outputs.detach()

            # Calculate discriminator loss.
            discriminator_loss = dis_criterion(discriminator(concat_outputs), discriminator(concat_hr))
            # Update discriminator loss count.
            running_dis_loss += discriminator_loss.item()

            generator_loss.backward()
            discriminator_loss.backward()

            gen_optimizer.step()
            dis_optimizer.step()

            # Update learning rate and loss to logging
            if summary_writer:
                if (i + 1) % every_n == 0:
                    summary_writer.add_scalar(GENERATOR_LOSS_NAME, running_gen_loss / total_images,
                                              global_step=minibatch_number)
                    summary_writer.add_scalar(DISCRIMINATOR_LOSS_NAME, running_dis_loss / total_images,
                                              global_step=minibatch_number)
                    summary_writer.add_scalar(SUPERVISED_LOSS_NAME, running_super_loss / total_images,
                                              global_step=minibatch_number)
                    minibatch_number += 1

            pbar.update(hr_images.shape[0])
        pbar.close()

        if summary_writer:
            learning_rate = next(iter(gen_optimizer.param_groups))["lr"]
            summary_writer.add_scalar(LEARNING_RATE_NAME, learning_rate,
                                      global_step=epoch)
        # Validate model
        with torch.no_grad():
            metric = validate(generator, validation_metric, validation_loader,
                              validation_transform, epoch, start_epoch, summary_writer, max_images)
            checkpoint_dict = {
                "epoch": epoch + 1,
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "gen_optimizer": gen_optimizer.state_dict(),
                "dis_optimizer": dis_optimizer.state_dict(),
                "best_metric": best_metric
            }
            if metric > best_metric:
                best_metric = metric
                torch.save(checkpoint_dict, CHECKPOINTS_PATH + f"{save_name}_Epoch{epoch:03}_Acc{metric:.5}.pth")


# def train_gan_paired(generator: nn.Module,
#                      discriminator: nn.Module,
#                      supervised_criterion,
#                      gen_criterion,
#                      dis_criterion,
#                      gen_optimizer: optim.Optimizer,
#                      dis_optimizer: optim.Optimizer,
#                      gan_loss_coeff: float,
#                      start_epoch: int,
#                      epochs: int,
#                      validation_metric,
#                      train_loader: DataLoader,
#                      validation_loader: DataLoader,
#                      gen_scheduler: List[float] = None,
#                      dis_scheduler: List[float] = None,
#                      gen_warmup: List[float] = None,
#                      dis_warmup: List[float] = None,
#                      summary_writer: SummaryWriter = None,
#                      max_images: int = 0,
#                      every_n: int = 1,
#                      best_metric: float = -1,
#                      save_name: str = ""):
#     minibatch_number = len(train_loader) * start_epoch // every_n
#
#     for epoch in range(start_epoch, epochs):
#         # If scheduler was passed, change lr to the one specified at each epoch.
#         if gen_scheduler:
#             for g in gen_optimizer.param_groups:
#                 g["lr"] = gen_scheduler[epoch]
#         if dis_scheduler:
#             for g in dis_optimizer.param_groups:
#                 g["lr"] = dis_scheduler[epoch]
#
#         generator.train()
#         discriminator.train()
#
#         # Zero all losses
#         running_gen_loss = 0
#         running_dis_loss = 0
#         running_super_loss = 0
#         total_images = 0
#
#         pbar = tqdm(total=len(train_loader.dataset))
#         for i, (lr_images, hr_images) in enumerate(train_loader):
#             # If warmup was passed, apply to mini_batch ONLY during the first epoch.
#             if gen_warmup:
#                 if i < len(gen_warmup):
#                     for g in gen_optimizer.param_groups:
#                         g["lr"] = gen_warmup[i]
#                 else:
#                     gen_warmup = None
#             if dis_warmup:
#                 if i < len(dis_warmup):
#                     for g in dis_optimizer.param_groups:
#                         g["lr"] = dis_warmup[i]
#                 else:
#                     dis_warmup = None
#
#             # Get proper hr minibatch and lr minibatch.
#             lr_images, hr_images = lr_images.to(DEVICE), hr_images.to(DEVICE)
#
#             # Update cumulatively iterated over images count.
#             total_images += hr_images.shape[0]
#
#             # Zero the optimizers
#             gen_optimizer.zero_grad()
#             dis_optimizer.zero_grad()
#
#             # Get model output
#             sr_images = generator(lr_images)
#
#             # Get concatenated images for conditional GAN
#             scaled_lr_images = F.interpolate(lr_images, scale_factor=2, mode="bicubic", align_corners=True)
#             concat_outputs = torch.cat((sr_images, scaled_lr_images), 1).to(DEVICE)
#             concat_hr = torch.cat((hr_images, scaled_lr_images), 1).to(DEVICE)
#
#             discriminator.requires_grad(False)
#             # Calculate supervised loss and total generator loss.
#             supervised_loss = supervised_criterion(sr_images, hr_images)
#             dis_out = discriminator(concat_outputs)
#             dis_hr = discriminator(concat_hr)
#             generator_loss = supervised_loss + \
#                              gan_loss_coeff * gen_criterion(dis_out, dis_hr)
#             # Update cumulative losses counts.
#             running_super_loss += supervised_loss.item()
#             running_gen_loss += generator_loss.item()
#
#             discriminator.requires_grad(True)
#             concat_outputs = concat_outputs.detach()
#
#             # Calculate discriminator loss.
#             discriminator_loss = dis_criterion(discriminator(concat_outputs), discriminator(concat_hr))
#             # Update discriminator loss count.
#             running_dis_loss += discriminator_loss.item()
#
#             generator_loss.backward()
#             discriminator_loss.backward()
#
#             gen_optimizer.step()
#             dis_optimizer.step()
#
#             # Update learning rate and loss to logging
#             if summary_writer:
#                 if (i + 1) % every_n == 0:
#                     summary_writer.add_scalar(GENERATOR_LOSS_NAME, running_gen_loss / total_images,
#                                               global_step=minibatch_number)
#                     summary_writer.add_scalar(DISCRIMINATOR_LOSS_NAME, running_dis_loss / total_images,
#                                               global_step=minibatch_number)
#                     summary_writer.add_scalar(SUPERVISED_LOSS_NAME, running_super_loss / total_images,
#                                               global_step=minibatch_number)
#                     minibatch_number += 1
#
#             pbar.update(hr_images.shape[0])
#         pbar.close()
#
#         if summary_writer:
#             learning_rate = next(iter(gen_optimizer.param_groups))["lr"]
#             summary_writer.add_scalar(LEARNING_RATE_NAME, learning_rate,
#                                       global_step=epoch)
#         # Validate model
#         with torch.no_grad():
#             metric = validate_paired(generator, validation_metric, validation_loader, epoch, start_epoch,
#                                      summary_writer,
#                                      max_images)
#             checkpoint_dict = {
#                 "epoch": epoch + 1,
#                 "generator": generator.state_dict(),
#                 "discriminator": discriminator.state_dict(),
#                 "gen_optimizer": gen_optimizer.state_dict(),
#                 "dis_optimizer": dis_optimizer.state_dict(),
#                 "best_metric": best_metric
#             }
#             if metric > best_metric:
#                 best_metric = metric
#                 torch.save(checkpoint_dict, CHECKPOINTS_PATH + f"{save_name}_Epoch{epoch:03}_Acc{metric:.5}.pth")
