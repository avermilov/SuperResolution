import argparse
import json

import lpips
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from models.discriminators import conv_discriminator, esrgan_discriminator
from models.generators import rdn, esrgan_generator
from scripts.losses import LSGANDisFakeLoss, LSGANDisRealLoss, LSGANGenLoss, VGGPerceptual
from scripts.metrics import PSNR, worker_init_fn, ssim
from scripts.training import train_gan
from scripts.transforms import get_train_lr_transform, get_validation_lr_transform, load_noises, load_kernels
from settings import DEVICE

if __name__ == "__main__":
    # Upscaling parameter
    SCALE = 2

    # Required arguments in the JSON file.
    REQ_ARGUMENTS = ["tr_path", "val_path", "epochs", "generator_lr", "discriminator_lr",
                     "train_batch_size", "validation_batch_size", "train_crop", "validation_crop",
                     "num_workers", "max_images_log", "gan_coeff", "every_n",
                     "discriminator_type", "generator_type", "supervised_loss_type", "save_name",
                     "metrics", "ker_path", "noises_path"]

    # Command line parser
    parser = argparse.ArgumentParser(description="Train a Super Resolution GAN.")
    parser.add_argument("--json", type=str, default=None,
                        help="JSON file with training arguments as parser.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint containing info for resuming training.")
    args = parser.parse_args()

    # Raise error if no JSON file passed.
    if not args.json:
        raise ValueError("No JSON file passed.")

    # Read arguments from JSON file.
    with open(args.json, "r") as file:
        data = json.load(file)

    # Search for missing required arguments
    missing_req = []
    for arg in REQ_ARGUMENTS:
        if arg not in data:
            missing_req.append(arg)

    # If any required argument is missing, raise error.
    if missing_req:
        raise ValueError(f"Some required arguments were not passed: {missing_req}")

    # Assign required arguments.
    tr_path = data["tr_path"]
    val_path = data["val_path"]
    epochs = data["epochs"]
    generator_lr_dict = data["generator_lr"]
    discriminator_lr = data["discriminator_lr"]
    train_batch_size = data["train_batch_size"]
    validation_batch_size = data["validation_batch_size"]
    train_crop = data["train_crop"]
    validation_crop = data["validation_crop"]
    num_workers = data["num_workers"]
    max_images_log = data["max_images_log"]
    gan_coeff = data["gan_coeff"]
    every_n = data["every_n"]
    discriminator_type = data["discriminator_type"]
    generator_type = data["generator_type"]
    supervised_loss_type = data["supervised_loss_type"]
    save_name = data["save_name"]
    metrics_names = data["metrics"]
    ker_path = data["ker_path"]
    noises_path = data["noises_path"]

    # Load noises and kernels
    load_noises(noises_path)
    load_kernels(ker_path)

    # Optional parameters
    best_metric = data["best_metric"] if "best_metric" in data else -1
    generator_warmup = data["generator_warmup"] if "generator_warmup" in data else None
    discriminator_warmup = data["discriminator_warmup"] if "discriminator_warmup" in data else None

    # Raise error if no metrics were passed
    if not metrics_names:
        raise ValueError("No validation metrics passed.")

    # Convert generator lr to proper list if dict was passed
    generator_lr = generator_lr_dict
    if isinstance(generator_lr_dict, dict):
        generator_lr = []
        for lr, count in generator_lr_dict.items():
            generator_lr += [float(lr)] * count

    # Use specified supervised_criterion
    supervised_criterion = None
    if supervised_loss_type == "VGGPerceptual":
        if "l1_coeff" not in data or "vgg_coeff" not in data:
            raise ValueError("Not all VGGPerceptual parameters were given.")
        supervised_criterion = VGGPerceptual(l1_coeff=data["l1_coeff"], vgg_coeff=data["vgg_coeff"])

    # Use specified discriminator
    discriminator = None
    if discriminator_type == "ConvDis":
        if "num_discriminator_features" not in data or "num_deep_layers" not in data:
            raise ValueError("Not all ConvDiscriminator parameters were given.")
        discriminator = conv_discriminator.ConvDiscriminator(num_channels=6,
                                                             num_features=data["num_discriminator_features"],
                                                             num_deep_layers=data["num_deep_layers"]).to(DEVICE)
    elif discriminator_type == "ESRDis":
        discriminator = esrgan_discriminator.Discriminator(num_channels=6).to(DEVICE)
    dis_optimizer = torch.optim.Adam(discriminator.parameters(),
                                     lr=discriminator_lr if isinstance(discriminator_lr, float) else
                                     discriminator_lr[0], betas=(0.5, 0.999))

    # Use specified generator
    generator = None
    if generator_type == "RDN":
        generator = rdn.RDN(SCALE, 3, 64, 64, 16, 8).to(DEVICE)
    elif generator_type == "ESRGen":
        generator = esrgan_generator.esrgan16(pretrained=False).to(DEVICE)
    gen_optimizer = torch.optim.Adam(generator.parameters(),
                                     lr=generator_lr if isinstance(generator_lr, float) else
                                     generator_lr[0], betas=(0.5, 0.999))

    # Use specified metrics for validation
    metrics = dict()
    metric_dict = {"psnr": PSNR(), "lpips_alex": lpips.LPIPS(net="alex").to(DEVICE),
                   "lpips_vgg": lpips.LPIPS(net="vgg").to(DEVICE), "ssim": ssim}
    for name, metric in metric_dict.items():
        if name in metrics_names:
            metrics[name] = metric

    gen_criterion = LSGANGenLoss()
    dis_fake_criterion = LSGANDisFakeLoss()
    dis_real_criterion = LSGANDisRealLoss()
    start_epoch = 0

    # Transform for converting image from training ImageFolder to tensor.
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomCrop(train_crop, train_crop)
    ])
    # Tranform for getting low res image from high res one.
    train_lr_transform = get_train_lr_transform(SCALE, train_crop)

    # Tranform for converting image from validation ImageFolder to tensor.
    validation_dataset_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.CenterCrop((validation_crop, validation_crop))
    ])

    # Transform for getting low res image from high res one for validation.
    validation_lr_transform = get_validation_lr_transform(SCALE, validation_crop)

    # Initialize datasets and data loaders.
    train_ds = torchvision.datasets.ImageFolder(tr_path, transform=train_transform)
    validation_ds = torchvision.datasets.ImageFolder(val_path, transform=validation_dataset_transform)

    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True,
                              num_workers=num_workers, worker_init_fn=worker_init_fn)
    validation_loader = DataLoader(validation_ds, batch_size=validation_batch_size,
                                   shuffle=False, num_workers=num_workers)

    sw = SummaryWriter()

    # If resume was requested, load all parameters from passed checkpoint file.
    if args.resume:
        checkpoint = torch.load(args.resume)
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        gen_optimizer.load_state_dict(checkpoint["gen_optimizer"])
        dis_optimizer.load_state_dict(checkpoint["dis_optimizer"])
        start_epoch = checkpoint["epoch"]
        best_metric = checkpoint["best_metric"]

    # elif expand_on:
    #     checkpoint = torch.load(expand_on)
    #     generator.load_state_dict(checkpoint["generator"])
    #     gen_optimizer.load_state_dict(checkpoint["gen_optimizer"])
    #     dis_optimizer.load_state_dict(checkpoint["dis_optimizer"])
    #     discriminator.load_state_dict(checkpoint["discriminator"])
    #     start_epoch = checkpoint["epoch"]
    #     max_images_log = checkpoint["max_images"]
    #     every_n = checkpoint["every_n"]
    #     gan_coeff = checkpoint["gan_coeff"]

    train_gan(generator=generator,
              discriminator=discriminator,
              supervised_criterion=supervised_criterion,
              gen_criterion=gen_criterion,
              dis_criterions=[dis_fake_criterion, dis_real_criterion],
              gen_optimizer=gen_optimizer,
              dis_optimizer=dis_optimizer,
              gan_loss_coeff=gan_coeff,
              start_epoch=start_epoch,
              epochs=epochs,
              metrics=metrics,
              train_loader=train_loader,
              validation_loader=validation_loader,
              lr_transform=train_lr_transform,
              validation_transform=validation_lr_transform,
              gen_scheduler=None if isinstance(generator_lr, float) else generator_lr,
              dis_scheduler=None if isinstance(discriminator_lr, float) else discriminator_lr,
              gen_warmup=generator_warmup,
              dis_warmup=discriminator_warmup,
              summary_writer=sw,
              max_images=max_images_log,
              every_n=every_n,
              best_metric=best_metric,
              save_name=save_name)
