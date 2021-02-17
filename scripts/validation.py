from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from settings import *


def validate(net: nn.Module,
             criterion,
             validation_loader: DataLoader,
             validation_transform,
             epoch: int,
             summary_writer: SummaryWriter = None,
             max_images: int = 0) -> float:
    net.eval()

    # Initialize values
    running_loss = 0
    images_logged = 0

    # Get resize
    shape = next(iter(validation_loader))[0].shape
    resize_transform = transforms.Resize((shape[2], shape[3]), interpolation=Image.BICUBIC)
    with torch.no_grad():
        for batch_no, hr_image in enumerate(validation_loader):
            # Get low res and high res images
            hr_image = hr_image[0]
            lr_image = validation_transform(hr_image)
            lr_image, hr_image = lr_image.to(DEVICE), hr_image.to(DEVICE)

            # Generate super res images and calculate loss
            sr_image = net(lr_image)
            loss = criterion(sr_image, hr_image)

            # Add calculated loss to running loss
            running_loss += loss.item()

            # If summary writer passed, log to tensorboard
            if summary_writer:
                # Transform and clamp images to [0, 1]
                for i in range(hr_image.shape[0]):
                    if images_logged < max_images:
                        if epoch == 0:
                            # Log bicubic resized image once only
                            resized_image = resize_transform(lr_image[i])
                            resized_image = (resized_image + 1) / 2
                            resized_image = torch.clamp(resized_image, min=0, max=1)
                            summary_writer.add_image(f"Validation/Image{images_logged:03}/LR Bicubic", resized_image,
                                                     global_step=epoch)

                            # Log high res image once only
                            hr = (hr_image[i] + 1) / 2
                            hr = torch.clamp(hr, min=0, max=1)
                            summary_writer.add_image(f"Validation/Image{images_logged:03}/HR", hr, global_step=epoch)

                        # Log super resolution image
                        sr = (sr_image[i] + 1) / 2
                        sr = torch.clamp(sr, min=0, max=1)
                        summary_writer.add_image(f"Validation/Image{images_logged:03}/LR SR", sr, global_step=epoch)

                        images_logged += 1
                    else:
                        break

    total_loss = running_loss / len(validation_loader)

    # If summary writer passed, log total loss to tensorboard
    if summary_writer:
        summary_writer.add_scalar(VALIDATION_NAME, total_loss, global_step=epoch)

    net.train()
    return total_loss
