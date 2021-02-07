import torch
from torch.utils.data import DataLoader
from skimage import io
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import RDN, ResNet, SimpleResBet, SRResNet
from train_net import train_net
from utils import *
from div2k_dataset import DIV2KDataset
from torchvision import transforms
import cv2
from models import *
from settings import *
from validate_net import validate
import cv2

if __name__ == "__main__":
    tfs = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])
    screwed_transform = transforms.Compose([
        # transforms.GaussianBlur(kernel_size=(3, 3), sigma=2),
        transforms.Resize((32, 32))
        # transforms.Resize((64, 64), interpolation=0)
    ])
    ds = DIV2KDataset(hr_folder="DIV2K_HR/", dcrop=(64, 64),
                      lr_transform=screwed_transform,
                      aug_transform=tfs, normalize=True)
    train_set, valid_set = torch.utils.data.random_split(ds, [800, 100])
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=10)
    valid_loader = DataLoader(valid_set, batch_size=4, shuffle=False, num_workers=10)

    # lr, hr = next(iter(train_loader))
    # lr, hr = lr.to(DEVICE), hr.to(DEVICE)
    # res = base_model(lr)
    # base_model = RDN(2, 3, 64, 64, 16, 8).to(DEVICE)
    # net = ResNet.sr_resnet18().to(DEVICE)
    net = RDN.RDN(2, 3, 64, 64, 16, 8).to(DEVICE)
    # net.load_state_dict(torch.load("results/epoch_22_acc_25.7.pth"))

    # img = torchvision.io.read_image("cyberpunkHR.jpg")
    # img = torch.unsqueeze(img, 0).to(DEVICE)
    # print(img.shape)
    # img = img.type(torch.FloatTensor).to(DEVICE)
    # res = net(img)[0]
    # torchvision.utils.save_image(torch.clamp(res, 0, 255) / 255, "cyberpunkHHR.jpg")
    # print(res.shape, torch.max(res), torch.min(res))
    # show_tensor_img((torch.squeeze(img, 0) - 128) / 128, permute=True)
    # show_tensor_img(res / 128 - 1, permute=True, clamp=True)

    # res = net(next(iter(valid_loader))[0].to(DEVICE))
    # print(res.shape)

    opt = torch.optim.Adam(net.parameters())
    # lr, hr = next(iter(valid_loader))
    # lr, hr = lr.cuda(), hr.cuda()
    # validate(net, PSNRmetrics, valid_loader)
    scheduler = [0.00033 * i for i in range(1, 4)] + [0.001] * 12 + [0.0001] * 10
    scheduler = [0.001] * 10 + [0.0002] * 5 + [0.0001] * 10

    train_net(net, 25, l1_and_vgg_loss, PSNR, opt, train_loader, valid_loader, scheduler)

    # d = set()
    # h1min, h2min = 1e10, 1e10
    # for hr, lr in tqdm(train_set):
    #     h1min = min(h1min, hr.shape[0])
    #     h2min = min(h2min, hr.shape[1])

    # print(h1min, h2min)
