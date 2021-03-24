import argparse
import json

import lpips
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from models.discriminators import conv_discriminator, esrgan_discriminator
from models.generators import rdn, newer_esrgan_generator
from scripts.losses import LSGANDisFakeLoss, LSGANDisRealLoss, LSGANGenLoss, VGGPerceptual
from scripts.metrics import PSNR, worker_init_fn, ssim
from scripts.training import train_gan
from scripts.transforms import get_train_lr_transform, get_validation_lr_transform, load_noises, load_kernels
from settings import DEVICE, set_device

if __name__ == "__main__":
    # Required arguments in the JSON file.
    REQ_ARGUMENTS = ["scale", "paths", "generator", "discriminator", "loaders", "loss", "logging", "device"]

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

    device = data["device"]
    set_device(device)

    # Assign required arguments.
    scale = data["scale"]
    # Check for x2 or x4 scale
    if scale != 2 and scale != 4:
        raise ValueError("Scale can only be 2 or 4.")

    epochs = data["epochs"]
    if epochs <= 0:
        raise ValueError("Epochs must be a positive integer.")

    paths_dict = data["paths"]
    kernels_path = paths_dict["kernels_path"]
    if kernels_path.lower() == "none":
        kernels_path = None
    noises_path = paths_dict["noises_path"]
    if noises_path.lower() == "none":
        noises_path = None
    tr_path = paths_dict["train_path"]
    valid_path = paths_dict["validation_path"]
    inference_source_path = paths_dict["inference_source_path"]
    if inference_source_path == "none":
        inference_source_path = None
    inference_results_path = paths_dict["inference_results_path"]
    if inference_results_path == "none":
        inference_results_path = None
    if inference_results_path is None and inference_source_path is not None \
            or inference_results_path is not None and inference_source_path is None:
        raise ValueError("Must pass both inference source and result paths or none.")

    generator_dict = data["generator"]
    generator_type = generator_dict["type"]
    generator_lr = generator_dict["learning_rate"]

    discriminator_dict = data["discriminator"]
    discriminator_type = discriminator_dict["type"]
    discriminator_lr = discriminator_dict["learning_rate"]

    loaders_dict = data["loaders"]
    train_crop = loaders_dict["train_crop"]
    validation_crop = loaders_dict["validation_crop"]
    train_batch_size = loaders_dict["train_batch_size"]
    validation_batch_size = loaders_dict["validation_batch_size"]
    num_workers = loaders_dict["num_workers"]
    inference_batch_size = loaders_dict["inference_batch_size"]
    if inference_batch_size == "none":
        inference_batch_size = None
    if inference_batch_size is None and inference_source_path is not None:
        raise ValueError("Must pass inference_batch_size if inference paths are specified.")

    loss_dict = data["loss"]
    loss_type = loss_dict["type"]
    supervised_coeff = loss_dict["supervised_coeff"]
    generator_coeff = loss_dict["generator_coeff"]
    if loss_type == "VGGPerceptual":
        if "l1_coeff" not in loss_dict or "vgg_coeff" not in loss_dict:
            raise ValueError("VGGPerceptual loss must specify L1 and VGG coefficients.")
        l1_coeff = loss_dict["l1_coeff"]
        vgg_coeff = loss_dict["vgg_coeff"]
    stepper = loss_dict["stepper"]
    if stepper == "none":
        stepper = []

    logging_dict = data["logging"]
    max_images = logging_dict["max_images"]
    save_prefix = logging_dict["save_prefix"]
    best_metric = logging_dict["best_metric"]
    if best_metric == "none":
        best_metric = None
    metrics_names = logging_dict["metrics"]

    # Load noises and kernels
    # todo: use 4x kernels
    load_noises(noises_path)
    load_kernels(kernels_path, scale)

    # Raise error if no metrics were passed
    if not metrics_names:
        raise ValueError("No validation metrics passed.")

    # Convert generator lr to proper list if dict or float was passed
    gen_lr = None
    if isinstance(generator_lr, dict):
        gen_lr = []
        total_count = 0
        for lr, count in generator_lr.items():
            gen_lr += [float(lr)] * count
            total_count += count
        if total_count != epochs:
            raise ValueError("Epochs and generator scheduler size must be equal.")
    elif isinstance(generator_lr, float):
        gen_lr = [generator_lr] * epochs

    # Convert discriminator lr to proper list if dict or float was passed
    dis_lr = None
    if isinstance(discriminator_lr, dict):
        dis_lr = []
        total_count = 0
        for lr, count in discriminator_lr.items():
            dis_lr += [float(lr)] * count
            total_count += count
        if total_count != epochs:
            raise ValueError("Epochs and discriminator scheduler size must be equal.")
    elif isinstance(discriminator_lr, float):
        dis_lr = [discriminator_lr] * epochs

    # Use specified supervised_criterion
    supervised_criterion = None
    if loss_type == "VGGPerceptual":
        supervised_criterion = VGGPerceptual(l1_coeff=l1_coeff, vgg_coeff=vgg_coeff)
    elif loss_type == "L1":
        supervised_criterion = nn.L1Loss()
    elif loss_type == "VGG":
        supervised_criterion = VGGPerceptual(l1_coeff=0, vgg_coeff=1)

    # Use specified discriminator
    discriminator = None
    if discriminator_type == "ConvDis":
        if "num_discriminator_features" not in discriminator_dict or \
                "num_deep_layers" not in discriminator_dict:
            raise ValueError("Not all ConvDiscriminator parameters were given.")
        num_features = discriminator_dict["num_discriminator_features"]
        num_deep_layers = discriminator_dict["num_deep_layers"].to(DEVICE)
        discriminator = conv_discriminator.ConvDiscriminator(num_channels=6,
                                                             num_features=num_features,
                                                             num_deep_layers=num_deep_layers)
    elif discriminator_type == "ESRDis":
        discriminator = esrgan_discriminator.Discriminator(num_channels=6).to(DEVICE)
    dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=dis_lr[0], betas=(0.5, 0.999))

    # Use specified generator
    generator = None
    if generator_type == "RDN":
        generator = rdn.RDN(scale, 3, 64, 64, 16, 8).to(DEVICE)
    elif generator_type == "ESRGen16":
        generator = newer_esrgan_generator.esrgan16(scale, pretrained=False).to(DEVICE)
    elif generator_type == "ESRGen23":
        generator = newer_esrgan_generator.esrgan23(scale, pretrained=False).to(DEVICE)
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=gen_lr[0], betas=(0.5, 0.999))

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
    train_lr_transform = get_train_lr_transform(scale, train_crop)

    # Tranform for converting image from validation ImageFolder to tensor.
    validation_dataset_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.CenterCrop((validation_crop, validation_crop))
    ])

    # Transform for getting low res image from high res one for validation.
    validation_lr_transform = get_validation_lr_transform(scale, validation_crop)

    # Transform for converting image from inference ImageFolder to tensor.
    inference_dataset_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Initialize datasets and data loaders.
    train_ds = torchvision.datasets.ImageFolder(tr_path, transform=train_transform)
    validation_ds = torchvision.datasets.ImageFolder(valid_path, transform=validation_dataset_transform)

    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True,
                              num_workers=num_workers, worker_init_fn=worker_init_fn)
    validation_loader = DataLoader(validation_ds, batch_size=validation_batch_size,
                                   shuffle=False, num_workers=num_workers)

    inference_loader = None
    if inference_source_path is not None:
        inference_ds = torchvision.datasets.ImageFolder(inference_source_path,
                                                        transform=inference_dataset_transform)
        inference_loader = DataLoader(inference_ds, batch_size=inference_batch_size, shuffle=False)

    sw = SummaryWriter()

    # If resume was requested, load all parameters from passed checkpoint file.
    if args.resume:
        checkpoint = torch.load(args.resume)
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        gen_optimizer.load_state_dict(checkpoint["gen_optimizer"])
        dis_optimizer.load_state_dict(checkpoint["dis_optimizer"])
        start_epoch = checkpoint["epoch"]

    train_gan(scale=scale,
              generator=generator,
              discriminator=discriminator,
              supervised_criterion=supervised_criterion,
              gen_criterion=gen_criterion,
              dis_criterions=[dis_fake_criterion, dis_real_criterion],
              gen_optimizer=gen_optimizer,
              dis_optimizer=dis_optimizer,
              supervised_coeff=supervised_coeff,
              gan_loss_coeff=generator_coeff,
              start_epoch=start_epoch,
              epochs=epochs,
              metrics=metrics,
              train_loader=train_loader,
              validation_loader=validation_loader,
              lr_transform=train_lr_transform,
              validation_transform=validation_lr_transform,
              gen_scheduler=gen_lr,
              dis_scheduler=dis_lr,
              summary_writer=sw,
              max_images=max_images,
              best_metric=best_metric,
              save_name=save_prefix,
              inference_loader=inference_loader,
              stepper=stepper)
