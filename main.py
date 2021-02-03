import torch
from torch.utils.data import DataLoader
from skimage import io
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from train_net import train_net
from utils import *
from div2k_dataset import DIV2KDataset
from torchvision import transforms
import cv2
from models import *
from settings import *
from validate_net import validate

if __name__ == "__main__":
    tfs = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])
    screwed_transform = transforms.Compose([transforms.GaussianBlur(kernel_size=(3, 3), sigma=2),
                                            transforms.Resize((32, 32)),
                                            transforms.Resize((64, 64), interpolation=0)])
    ds = DIV2KDataset(hr_folder="DIV2K_HR/", dcrop=(64, 64),
                      lr_transform=screwed_transform,
                      aug_transform=tfs, normalize=True)
    train_set, valid_set = torch.utils.data.random_split(ds, [800, 100])
    train_loader = DataLoader(train_set, batch_size=10, shuffle=True, num_workers=10)
    valid_loader = DataLoader(valid_set, batch_size=10, shuffle=False)
    # for lr, hr in valid_loader:
    #     print(lr.shape, hr.shape)
    # lr, hr = next(iter(train_loader))
    # sw = SummaryWriter()
    # show_tensor_img(lr[0])
    # sw.add_image("img", lr[0] / 2 + 0.5)
    # sw.add_image("testing", create_grid(lr[0], hr[0], hr[0], normalized=True))
    # sw.close()
    # exit()
    # lr, hr = next(iter(train_loader))
    # for i in range(10):
    #     show_tensor_img(lr[i])
    #     show_tensor_img(hr[i])
    # exit()

    res_net = torchvision.models.resnet18(pretrained=True)

    res_net.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    # base_model = nn.Sequential(
    #     res_net.conv1,
    #     res_net.bn1,
    #     res_net.relu,
    #     nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False),
    #     res_net.layer1,
    #     nn.Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # ).to(DEVICE)
    base_model = IgnatNet(res_net).to(DEVICE)

    # lr, hr = next(iter(train_loader))
    # lr, hr = lr.to(DEVICE), hr.to(DEVICE)
    # res = base_model(lr)

    opt = torch.optim.Adam(base_model.parameters())

    net = SRResNet(scaling_factor=2).to(DEVICE)
    # lr, hr = next(iter(valid_loader))
    # lr, hr = lr.cuda(), hr.cuda()
    # validate(net, PSNRmetrics, valid_loader)

    train_net(base_model, 10, compute_loss, PSNRmetrics, opt, train_loader, valid_loader)

    # d = set()
    # h1min, h2min = 1e10, 1e10
    # for hr, lr in tqdm(train_set):
    #     h1min = min(h1min, hr.shape[0])
    #     h2min = min(h2min, hr.shape[1])

    # print(h1min, h2min)
