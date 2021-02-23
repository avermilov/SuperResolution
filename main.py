import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from scripts.losses import LSGANDisLoss, LSGANGenLoss, VGGPerceptual
from scripts.training import train_gan
from models import rdn
from settings import DEVICE
from models.conv_discriminator import ConvDiscriminator
from scripts.metrics import PSNR, worker_init_fn
from scripts.argparser import parser
import json

if __name__ == "__main__":
    # Upscaling parameter
    SCALE = 2

    # Get all parameters from command line parser.
    args = parser.parse_args()
    tr_path = args.tr_path
    val_path = args.val_path
    epochs = args.epochs
    generator_lr = args.generator_lr
    discriminator_lr = args.discriminator_lr
    train_batch_size = args.train_batch_size
    validation_batch_size = args.validation_batch_size
    train_crop = args.train_crop
    validation_crop = args.validation_crop
    resume_path = args.resume_path
    discriminator_num_features = args.discriminator_num_features
    num_workers = args.num_workers
    l1_coeff = args.l1_coeff
    vgg_coeff = args.vgg_coeff
    gan_coeff = args.gan_coeff
    max_images_log = args.max_images_log
    every_n = args.every_n
    scheduler = args.scheduler
    warmup = args.warmup
    best_metric = args.best_metric

    # If a JSON was passed, override any value encountered with one from file.
    if args.json:
        with open(args.json, "r") as file:
            data = json.load(file)
        if "tr_path" in data: tr_path = data["tr_path"]
        if "val_path" in data: val_path = data["val_path"]
        if "epochs" in data: epochs = data["epochs"]
        if "generator_lr" in data: generator_lr = data["generator_lr"]
        if "discriminator_lr" in data: discriminator_lr = data["discriminator_lr"]
        if "train_batch_size" in data: train_batch_size = data["train_batch_size"]
        if "validation_batch_size" in data: validation_batch_size = data["validation_batch_size"]
        if "train_crop" in data: train_crop = data["train_crop"]
        if "validation_crop" in data: validation_crop = data["validation_crop"]
        if "resume_path" in data: resume_path = data["resume_path"]
        if "discriminator_num_features" in data: discriminator_num_features = data["discriminator_num_features"]
        if "num_workers" in data: num_workers = data["num_workers"]
        if "l1_coeff" in data: l1_coeff = data["l1_coeff"]
        if "vgg_coeff" in data: vgg_coeff = data["vgg_coeff"]
        if "gan_coeff" in data: gan_coeff = data["gan_coeff"]
        if "max_images_log" in data: max_images_log = data["max_images_log"]
        if "every_n" in data: every_n = data["every_n"]
        if "scheduler" in data: scheduler = data["scheduler"]
        if "warmup" in data: warmup = data["warmup"]
        if "best_metric" in data: best_metric = data["best_metric"]

    # Transform for converting image from training ImageFolder to tensor.
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomCrop(train_crop, train_crop)
    ])
    # Tranform for getting low res image from high res one.
    lr_transform = transforms.Compose([
        transforms.Resize((train_crop // SCALE, train_crop // SCALE),
                          interpolation=Image.BICUBIC)
    ])
    # Tranform for converting image from validation ImageFolder to tensor.
    validation_dataset_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.CenterCrop((validation_crop, validation_crop))
    ])
    # Transform for getting low res image from high res one for validation.
    validation_transform = transforms.Compose([
        transforms.Resize((validation_crop // SCALE, validation_crop // SCALE),
                          interpolation=Image.BICUBIC)
    ])

    # Initialize datasets and data loaders.
    train_ds = torchvision.datasets.ImageFolder(tr_path, transform=train_transform)
    validation_ds = torchvision.datasets.ImageFolder(val_path, transform=validation_dataset_transform)

    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True,
                              num_workers=num_workers, worker_init_fn=worker_init_fn)
    validation_loader = DataLoader(validation_ds, batch_size=validation_batch_size,
                                   shuffle=False, num_workers=num_workers)

    sw = SummaryWriter()

    # General training parameters.
    generator = rdn.RDN(SCALE, 3, 64, 64, 16, 8).to(DEVICE)
    discriminator = ConvDiscriminator(num_channels=6, num_features=discriminator_num_features).to(DEVICE)
    supervised_criterion = VGGPerceptual(l1_coeff=l1_coeff, vgg_coeff=vgg_coeff)
    gen_criterion = LSGANGenLoss()
    dis_criterion = LSGANDisLoss()
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=generator_lr, betas=(0.5, 0.999))
    dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=discriminator_lr, betas=(0.5, 0.999))
    start_epoch = 0
    epochs = epochs
    validation_metric = PSNR()

    # If resume was requested, load all parameters from passed checkpoint file.
    if resume_path:
        checkpoint = torch.load(resume_path)
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        gen_optimizer.load_state_dict(checkpoint["gen_optimizer"])
        dis_optimizer.load_state_dict(checkpoint["dis_optimizer"])
        warmup = checkpoint["warmup"]
        scheduler = checkpoint["scheduler"]
        start_epoch = checkpoint["epoch"]
        epochs = checkpoint["epochs"]
        max_images_log = checkpoint["max_images"]
        every_n = checkpoint["every_n"]
        gan_coeff = checkpoint["gan_coeff"]
        best_metric = checkpoint["best_metric"]

    train_gan(generator=generator,
              discriminator=discriminator,
              supervised_criterion=supervised_criterion,
              gen_criterion=gen_criterion,
              dis_criterion=dis_criterion,
              gen_optimizer=gen_optimizer,
              dis_optimizer=dis_optimizer,
              gan_loss_coeff=gan_coeff,
              start_epoch=start_epoch,
              epochs=epochs,
              validation_metric=validation_metric,
              train_loader=train_loader,
              validation_loader=validation_loader,
              lr_transform=lr_transform,
              validation_transform=validation_transform,
              scheduler=scheduler,
              warmup=warmup,
              summary_writer=sw,
              max_images=max_images_log,
              every_n=every_n,
              best_metric=best_metric)
