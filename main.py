import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import validate_net
from predict import predict
from train_net import train_net
from models import RDN
from settings import *
from algorithms import *

if __name__ == "__main__":
    # Transforms
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomCrop(64, 64)])
    lr_transform = transforms.Compose([
        transforms.Resize((32, 32))
    ])
    predict_transform = transforms.Compose([
        transforms.Resize((150, 150))
    ])

    # Dataset and loaders
    ds = torchvision.datasets.ImageFolder(DATASET_PATH, transform=data_transform)
    train_set, valid_set = torch.utils.data.random_split(ds, [800, 100])
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_set, batch_size=4, shuffle=False, num_workers=8)
    predict_ds = torchvision.datasets.ImageFolder(PREDICT_FROM_PATH, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    predict_loader = DataLoader(predict_ds, shuffle=False, batch_size=1)

    # Trajn specifics
    scheduler = [1e-3] * 8 + [1e-4] * 12 + [1e-5] * 10  # + [0.0002] * 5 + [0.0001] * 10
    epochs = 30
    net = RDN.RDN(2, 3, 64, 64, 16, 8).to(DEVICE)
    opt = torch.optim.Adam(net.parameters())
    train_crit = l1_and_vgg_loss
    valid_crit = psnr

    # sw = SummaryWriter()
    # predict(net, 0, predict_loader, predict_transform, sw)
    # sw.close()
    # exit()
    train_net(net, epochs, train_crit, valid_crit, opt, train_loader,
              valid_loader, predict_loader, lr_transform, predict_transform, scheduler)
