import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from settings import DEVICE, INFERENCE_PATH


def inference(net: nn.Module, epoch: int, inference_loader: DataLoader, summary_writer: SummaryWriter) -> None:
    net.eval()
    count = 0
    with torch.no_grad():
        pbar = tqdm(total=len(inference_loader.dataset))
        for (images, _) in inference_loader:
            images = images.to(DEVICE)
            sr_images = net(images)
            sr_images = (sr_images + 1) / 2
            sr_images = torch.clamp(sr_images, 0, 1)
            for i in range(images.shape[0]):
                sr_path = INFERENCE_PATH + f"Image {count:05}/"
                summary_writer.add_image(sr_path, sr_images[i], global_step=epoch)
                count += 1
                pbar.update(1)
        pbar.close()
    net.train()
