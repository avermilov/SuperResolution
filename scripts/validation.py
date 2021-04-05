from collections import defaultdict

from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from typing import List
from tqdm import tqdm

from settings import *


def validate(net: nn.Module,
             metrics,
             validation_loader: DataLoader,
             validation_transform,
             epoch: int,
             start_epoch: int,
             summary_writer: SummaryWriter = None,
             max_images: int = 0) -> List[float]:
    net.eval()

    # Initialize values
    images_logged = 0

    running_losses = defaultdict(int)

    # Get resize
    shape = next(iter(validation_loader))[0].shape
    resize_transform = transforms.Resize((shape[2], shape[3]), interpolation=Image.BICUBIC)
    with torch.no_grad():
        pbar = tqdm(total=len(validation_loader.dataset))
        for batch_no, hr_image in enumerate(validation_loader):
            # Get low res and high res images
            hr_image = hr_image[0]
            lr_image = validation_transform(hr_image)
            lr_image, hr_image = lr_image.to(DEVICE), hr_image.to(DEVICE)

            # Generate super res images and calculate loss
            sr_image = net(lr_image)

            for name, metric in metrics.items():
                if name == "ssim":
                    sr_image_normalized = (sr_image + 1) / 2
                    hr_image_normalized = (hr_image + 1) / 2
                    loss = metric(sr_image_normalized, hr_image_normalized)
                    running_losses[name] += loss.item()
                elif name == "psnr":
                    loss = metric(sr_image, hr_image)
                    running_losses[name] += loss.item()
                else:
                    loss = metric(sr_image, hr_image)
                    running_losses[name] += torch.mean(loss).item()

            # If summary writer passed, log to tensorboard
            if summary_writer:
                # Transform and clamp images to [0, 1]
                for i in range(hr_image.shape[0]):
                    if images_logged < max_images:
                        if epoch - start_epoch <= 1:
                            # Log high res image once only
                            hr = (hr_image[i] + 1) / 2
                            hr = torch.clamp(hr, min=0, max=1)
                            summary_writer.add_image(VALIDATION_PATH + f"Image{images_logged:03}/HR", hr,
                                                     global_step=epoch)

                        # Log bicubic resized image once only
                        resized_image = resize_transform(lr_image[i])
                        resized_image = (resized_image + 1) / 2
                        resized_image = torch.clamp(resized_image, min=0, max=1)
                        summary_writer.add_image(VALIDATION_PATH + f"Image{images_logged:03}/LR BI", resized_image,
                                                 global_step=epoch)
                        # Log super resolution image
                        sr = (sr_image[i] + 1) / 2
                        sr = torch.clamp(sr, min=0, max=1)
                        summary_writer.add_image(VALIDATION_PATH + f"Image{images_logged:03}/LR SR", sr,
                                                 global_step=epoch)

                        images_logged += 1
                    else:
                        break

            pbar.update(lr_image.shape[0])
    pbar.close()

    # If summary writer passed, log total losses to tensorboard
    total_losses = []
    if summary_writer:
        for name, run_loss in running_losses.items():
            total_loss = run_loss / len(validation_loader)
            total_losses.append(total_loss)
            summary_writer.add_scalar(METRIC_NAMES[name], total_loss, global_step=epoch)

    net.train()
    return total_losses
