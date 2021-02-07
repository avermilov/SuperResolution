import io
import torch
import matplotlib.pyplot as plt
from torch import nn
import torchvision
import torch.nn.functional as F
import cv2
from torchvision import transforms


def l1_and_vgg_loss(x: torch.tensor, y: torch.tensor, model: nn.Module, validation_model: nn.Module) -> float:
    y_pred = model(x)
    l1 = nn.L1Loss()(y_pred, y)
    l1_features = nn.L1Loss()(validation_model(y_pred), validation_model(y))
    return l1 + l1_features


def l1_loss(x: torch.tensor, y: torch.tensor, model: nn.Module, validation_model: nn.Module) -> float:
    y_pred = model(x)
    l1 = nn.L1Loss()(y_pred, y)
    return l1


def PSNR(x, y, model):
    y_pred = model(x)
    mse = nn.MSELoss()(y_pred, y)
    return 10 * torch.log10(1 / mse)


def show_tensor_img(img: torch.tensor, permute=False, clamp=False) -> None:
    with torch.no_grad():
        img = (img / 2 + 0.5).detach().cpu()
        if permute:
            img = img.permute(1, 2, 0)
        if clamp:
            img = torch.clamp(img, 0, 1)
        plt.imshow(img)
        plt.show()


def create_grid(lr_img: torch.tensor, result: torch.tensor, hr_img: torch.tensor,
                normalized: bool = False) -> torch.tensor:
    lr, res, hr = lr_img, result, hr_img
    if normalized:
        lr = lr_img / 2 + 0.5
        hr = hr_img / 2 + 0.5
        res = result / 2 + 0.5
    zeros = torch.zeros(3, lr_img.shape[1], lr_img.shape[1])
    fit = torch.cat((zeros, zeros), dim=1)
    fit2 = torch.cat((zeros, lr), dim=1)
    # ts = torch.cat((torch.cat((fit, fit2), dim=2), res, hr), dim=2)
    lr = torch.cat((fit, fit2), dim=2)
    ts = torch.cat((lr, res, hr), dim=2)
    return torchvision.utils.make_grid(ts)


def get_imgs(ind: int) -> (torch.tensor, torch.tensor):
    lr = torch.from_numpy(io.imread(f"DIV2K_LR_bicubic/X2/{ind:04}x2.png")).permute(2, 0, 1).type(torch.FloatTensor)
    lr = transforms.Normalize((128, 128, 128), (128, 128, 128))(lr)

    hr = torch.from_numpy(io.imread(f"DIV2K_HR/{ind:04}.png")).permute(2, 0, 1).type(torch.FloatTensor)
    hr = transforms.Normalize((128, 128, 128), (128, 128, 128))(hr)
    return lr, hr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
