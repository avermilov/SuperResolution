from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from scripts.training import train
from models import rdn
from settings import *
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

    # Trajn specifics
    scheduler = [1e-4] * 20 + [5e-5] * 20 + [1e-4] * 20  # + [0.0002] * 5 + [0.0001] * 10
    warmup = [1e-5, 4e-5, 8e-5]
    epochs = 20
    net = rdn.RDN(2, 3, 64, 64, 16, 8).to(DEVICE)
    opt = torch.optim.Adam(net.parameters())
    train_crit = VGGPerceptual()
    valid_crit = PSNR()

    sw = SummaryWriter()

    train(net=net,
          epochs=epochs,
          train_criterion=train_crit,
          validation_criterion=valid_crit,
          optimizer=opt,
          train_loader=train_loader,
          validation_loader=validation_loader,
          lr_transform=lr_transform,
          validation_transform=validation_transform,
          scheduler=scheduler,
          warmup=warmup,
          summary_writer=sw,
          max_images=MAX_IMAGES_LOG)
