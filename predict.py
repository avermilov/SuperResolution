import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from settings import predict_loader, PREDICT_TO_PATH


def predict(model: nn.Module, epoch: int, sw: SummaryWriter = None) -> None:
    model.eval()
    with torch.no_grad():
        for i, img in enumerate(predict_loader):
            res = model(img[0].cuda())[0]
            res = torch.clamp(res / 2 + 0.5, 0, 1)
            if sw:
                sw.add_image(f"img{i}_val", res, global_step=epoch)
            torchvision.utils.save_image(res, f"{PREDICT_TO_PATH}/img{i}_epoch{epoch}.jpg")
    model.train()
