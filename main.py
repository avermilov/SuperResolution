import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from scripts.losses import LSGANDisLoss, LSGANGenLoss, VGGPerceptual
from scripts.training import train_gan
from models import rdn
from settings import DEVICE, CHECKPOINTS_PATH
from models.conv_discriminator import ConvDiscriminator
from scripts.metrics import PSNR, worker_init_fn
from scripts.argparser import parser

if __name__ == "__main__":
    args = parser.parse_args()
    SCALE = 2

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomCrop(args.train_crop, args.train_crop)
    ])
    lr_transform = transforms.Compose([
        transforms.Resize((args.train_crop // SCALE, args.train_crop // SCALE),
                          interpolation=Image.BICUBIC)
    ])
    validation_dataset_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.CenterCrop((args.validation_crop, args.validation_crop))
    ])
    validation_transform = transforms.Compose([
        transforms.Resize((args.validation_crop // SCALE, args.validation_crop // SCALE),
                          interpolation=Image.BICUBIC)
    ])

    train_ds = torchvision.datasets.ImageFolder(args.tr_path, transform=train_transform)
    validation_ds = torchvision.datasets.ImageFolder(args.val_path, transform=validation_dataset_transform)

    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True,
                              num_workers=args.num_workers, worker_init_fn=worker_init_fn)
    validation_loader = DataLoader(validation_ds, batch_size=args.validation_batch_size,
                                   shuffle=False, num_workers=args.num_workers)

    sw = SummaryWriter()

    generator = rdn.RDN(2, 3, 64, 64, 16, 8).to(DEVICE)
    discriminator = ConvDiscriminator(num_channels=6, num_features=args.discriminator_num_features).to(DEVICE)
    supervised_criterion = VGGPerceptual(l1_coeff=args.l1_coeff, vgg_coeff=args.vgg_coeff)
    gen_criterion = LSGANGenLoss()
    dis_criterion = LSGANDisLoss()
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=args.generator_lr, betas=(0.5, 0.999))
    dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.discriminator_lr, betas=(0.5, 0.999))
    start_epoch = 0
    epochs = args.epochs
    validation_metric = PSNR()
    scheduler = [1e-3] * 4
    warmup = [1e-4, 4e-4, 8e-4]

    if args.resume_path:
        checkpoint = torch.load(CHECKPOINTS_PATH + args.resume_path)
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        gen_optimizer.load_state_dict(checkpoint["gen_optimizer"])
        dis_optimizer.load_state_dict(checkpoint["dis_optimizer"])
        start_epoch = checkpoint["epoch"]
        epochs = checkpoint["epochs"]

    train_gan(generator=generator,
              discriminator=discriminator,
              supervised_criterion=supervised_criterion,
              gen_criterion=gen_criterion,
              dis_criterion=dis_criterion,
              gen_optimizer=gen_optimizer,
              dis_optimizer=dis_optimizer,
              gan_loss_coeff=0.1,
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
              max_images=args.max_images_log,
              every_n=args.every_n)
