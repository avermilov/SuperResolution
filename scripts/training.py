from typing import List

import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from scripts.inference import inference
from scripts.validation import validate
from settings import *


def train_gan(scale: int,
              generator: nn.Module,
              discriminator: nn.Module,
              supervised_criterion,
              gen_criterion,
              dis_criterions,
              gen_optimizer: optim.Optimizer,
              dis_optimizer: optim.Optimizer,
              supervised_coeff: float,
              gan_loss_coeff: float,
              start_epoch: int,
              epochs: int,
              metrics,
              train_loader: DataLoader,
              validation_loader: DataLoader,
              lr_transform: transforms.Compose,
              validation_transform: transforms.Compose,
              gen_scheduler: List[float] = None,
              dis_scheduler: List[float] = None,
              summary_writer: SummaryWriter = None,
              max_images=0,
              best_metric=-1,
              save_name: str = "",
              inference_loader: DataLoader = None,
              stepper=None):
    dis_fake_criterion, dis_real_criterion = dis_criterions

    save_every = best_metric == "every"
    if best_metric == "none":
        best_metric = -10 ** 9

    best_metric_split = best_metric.split()
    every_n = None
    if len(best_metric_split) > 1:
        every_n = int(best_metric_split[1])

    use_stepper = stepper is not None
    if use_stepper:
        discriminator_loss = 1
        stepper_off_treshold = stepper["off_threshold"]
        stepper_on_threshold = stepper["on_threshold"]
    stepper_active = False

    total_minibatches = len(train_loader)
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
        running_dis_fake_loss = 0
        running_dis_real_loss = 0
        running_dis_total_loss = 0
        running_super_loss = 0
        running_gen_total_loss = 0

        pbar = tqdm(total=len(train_loader.dataset))
        for i, hr_images in enumerate(train_loader):
            # Get proper hr minibatch and lr minibatch.
            hr_images = hr_images[0]
            lr_images = lr_transform(hr_images)
            lr_images, hr_images = lr_images.to(DEVICE), hr_images.to(DEVICE)

            # Zero the optimizers
            gen_optimizer.zero_grad()
            dis_optimizer.zero_grad()

            # Get model output
            sr_images = generator(lr_images)

            # Get concatenated images for conditional GAN
            scaled_lr_images = F.interpolate(lr_images, scale_factor=scale, mode="bicubic", align_corners=True)
            concat_outputs = torch.cat((sr_images, scaled_lr_images), 1).to(DEVICE)
            concat_hr = torch.cat((hr_images, scaled_lr_images), 1).to(DEVICE)

            discriminator.requires_grad(False)

            # Calculate supervised loss and total generator loss.
            supervised_loss = supervised_coeff * supervised_criterion(sr_images, hr_images)
            dis_out = discriminator(concat_outputs)
            dis_hr = discriminator(concat_hr)
            gen_loss = gan_loss_coeff * gen_criterion(dis_out, dis_hr)
            generator_total_loss = supervised_loss + gen_loss

            # Update cumulative losses counts.
            running_super_loss += supervised_loss.item()
            running_gen_loss += gen_loss.item()
            running_gen_total_loss += generator_total_loss.item()

            if use_stepper:
                if not stepper_active and discriminator_loss < stepper_off_treshold:
                    stepper_active = True
                elif stepper_active and discriminator_loss > stepper_on_threshold:
                    stepper_active = False

            if not stepper_active or not use_stepper:
                discriminator.requires_grad(True)

            concat_outputs = concat_outputs.detach()

            # Calculate discriminator loss.
            dis_outputs_score = discriminator(concat_outputs)
            dis_hr_score = discriminator(concat_hr)
            dis_fake_loss = dis_fake_criterion(dis_outputs_score)
            dis_real_loss = dis_real_criterion(dis_hr_score)
            discriminator_loss = dis_fake_loss + dis_real_loss

            # Update discriminator loss count.
            running_dis_fake_loss += dis_fake_loss.item()
            running_dis_real_loss += dis_real_loss.item()
            running_dis_total_loss += discriminator_loss.item()

            generator_total_loss.backward()
            gen_optimizer.step()

            if not stepper_active or not use_stepper:
                discriminator_loss.backward()
                dis_optimizer.step()

            pbar.update(hr_images.shape[0])
        pbar.close()

        if summary_writer:
            learning_rate = next(iter(gen_optimizer.param_groups))["lr"]
            summary_writer.add_scalar(LEARNING_RATE_NAME, learning_rate,
                                      global_step=epoch)
            summary_writer.add_scalar(GENERATOR_LOSS_NAME, running_gen_loss / total_minibatches,
                                      global_step=epoch)
            summary_writer.add_scalar(DISCRIMINATOR_TOTAL_LOSS_NAME, running_dis_total_loss / total_minibatches,
                                      global_step=epoch)
            summary_writer.add_scalar(DISCRIMINATOR_FAKE_LOSS_NAME, running_dis_fake_loss / total_minibatches,
                                      global_step=epoch)
            summary_writer.add_scalar(DISCRIMINATOR_REAL_LOSS_NAME, running_dis_real_loss / total_minibatches,
                                      global_step=epoch)
            summary_writer.add_scalar(SUPERVISED_LOSS_NAME, running_super_loss / total_minibatches,
                                      global_step=epoch)
            summary_writer.add_scalar(GENERATOR_TOTAL_LOSS_NAME, running_gen_total_loss / total_minibatches,
                                      global_step=epoch)

        # Validate model
        with torch.no_grad():
            metric = validate(generator, metrics, validation_loader,
                              validation_transform, epoch, start_epoch, summary_writer, max_images)
            checkpoint_dict = {
                "epoch": epoch + 1,
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "gen_optimizer": gen_optimizer.state_dict(),
                "dis_optimizer": dis_optimizer.state_dict(),
                "best_metric": best_metric
            }
            if every_n is not None:
                if (epoch - start_epoch) % every_n == every_n - 1:
                    torch.save(checkpoint_dict,
                               CHECKPOINTS_PATH + f"{save_name}_Epoch{epoch:03}_Metric{metric[0]:.5}.pth")
            elif save_every or epoch == epochs - 1:
                torch.save(checkpoint_dict, CHECKPOINTS_PATH + f"{save_name}_Epoch{epoch:03}_Metric{metric[0]:.5}.pth")
            elif metric[0] > best_metric:
                best_metric = metric[0]
                torch.save(checkpoint_dict, CHECKPOINTS_PATH + f"{save_name}_Epoch{epoch:03}_Metric{metric[0]:.5}.pth")

            if inference_loader is not None:
                inference(generator, epoch, inference_loader, summary_writer)
