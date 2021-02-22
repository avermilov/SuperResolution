from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from scripts.training import train, train_gan
from models import rdn
from settings import *
from models.conv_discriminator import ConvDiscriminator
from scripts.algorithms import *

if __name__ == "__main__":
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomCrop(64, 64)
    ])
    lr_transform = transforms.Compose([
        transforms.Resize((32, 32), interpolation=Image.BICUBIC)
    ])
    validation_dataset_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.CenterCrop((256, 256))
    ])
    validation_transform = transforms.Compose([
        transforms.Resize((128, 128), interpolation=Image.BICUBIC)
    ])

    train_ds = torchvision.datasets.ImageFolder(TRAIN_PATH, transform=train_transform)
    validation_ds = torchvision.datasets.ImageFolder(VALIDATION_PATH, transform=validation_dataset_transform)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=10, worker_init_fn=worker_init_fn)
    validation_loader = DataLoader(validation_ds, batch_size=8, shuffle=False, num_workers=10)

    sw = SummaryWriter()

    # Trajn specifics
    # scheduler = [1e-4] * 20 + [5e-5] * 20 + [1e-4] * 20  # + [0.0002] * 5 + [0.0001] * 10
    # warmup = [1e-5, 4e-5, 8e-5]
    # epochs = 20
    # net = rdn.RDN(2, 3, 64, 64, 16, 8).to(DEVICE)
    # opt = torch.optim.Adam(net.parameters())
    # train_crit = VGGPerceptual()
    # valid_crit = PSNR()
    #
    #
    #
    # train(net=net,
    #       epochs=epochs,
    #       train_criterion=train_crit,
    #       validation_criterion=valid_crit,
    #       optimizer=opt,
    #       train_loader=train_loader,
    #       validation_loader=validation_loader,
    #       lr_transform=lr_transform,
    #       validation_transform=validation_transform,
    #       scheduler=scheduler,
    #       warmup=warmup,
    #       summary_writer=sw,
    #       max_images=MAX_IMAGES_LOG)

    generator = rdn.RDN(2, 3, 64, 64, 16, 8).to(DEVICE)
    discriminator = ConvDiscriminator(num_channels=6, num_features=64).to(DEVICE)
    supervised_criterion = VGGPerceptual()
    gen_criterion = LSGANGenLoss()
    dis_criterion = LSGANDisLoss()
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3, betas=(0.5, 0.999))
    dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    epochs = 2
    validation_metric = PSNR()
    scheduler = [1e-3] * 2

    # checkpoint = torch.load(CHECKPOINTS_PATH)
    # generator.load_state_dict(checkpoint["generator"])
    # discriminator.load_state_dict(checkpoint["discriminator"])
    # gen_optimizer.load_state_dict(checkpoint["gen_optimizer"])
    # dis_optimizer.load_state_dict(checkpoint["dis_optimizer"])
    # epoch = checkpoint["epoch"]
    train_gan(generator=generator,
              discriminator=discriminator,
              supervised_criterion=supervised_criterion,
              gen_criterion=gen_criterion,
              dis_criterion=dis_criterion,
              gen_optimizer=gen_optimizer,
              dis_optimizer=dis_optimizer,
              gan_loss_coeff=0.1,
              epochs=2,
              validation_metric=validation_metric,
              train_loader=train_loader,
              validation_loader=validation_loader,
              lr_transform=lr_transform,
              validation_transform=validation_transform,
              scheduler=scheduler,
              warmup=[1e-4, 4e-4, 8e-4],
              summary_writer=sw,
              max_images=10)
