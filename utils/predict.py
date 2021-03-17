from argparse import ArgumentParser

import PIL
import torch
import torchvision

from settings import DEVICE
from models.generators import *
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import json

from models.generators import rdn, esrgan_generator, resnet, sr_resnet, simple_resnet

argparser = ArgumentParser()
argparser.add_argument("json", type=str, default=None)
argparser.add_argument("--img_path", type=str, default=None)
argparser.add_argument("--res_path", type=str, default=None)
argparser.add_argument("--net_type", type=str, default=None)
argparser.add_argument("--net_path", type=str, default=None)
args = argparser.parse_args()

with open(args.json, "r") as file:
    data = json.load(file)

    img_path = data["img_path"]
    res_path = data["res_path"]
    net_path = data["net_path"]
    net_type = data["net_type"]
    save_prefix = data["save_prefix"]
    save_bicubic = data["save_bicubic"]

ds_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

nets = {"rdn": rdn.RDN(2, 3, 64, 64, 16, 8),
        "esrgen16": esrgan_generator.esrgan16(),
        "esrgen23": esrgan_generator.esrgan23()}
net = nets[net_type.lower()].to(DEVICE)
gen_dict = torch.load(net_path)["generator"]
net.load_state_dict(gen_dict)
net.eval()

ds = ImageFolder(img_path, transform=ds_transform)
loader = DataLoader(ds, batch_size=1, shuffle=False)

with torch.no_grad():
    for i, (img, _) in enumerate(loader):
        img = img.to(DEVICE)
        sr = net(img)[0].to(DEVICE)
        sr = (sr + 1) / 2
        sr = torch.clamp(sr, 0, 1)
        torchvision.utils.save_image(sr, res_path + f"{save_prefix}_sr{i:06}.png")
        if save_bicubic:
            h, w = img.shape[-2:]
            bic = transforms.Resize((2 * h, 2 * w), interpolation=PIL.Image.BICUBIC)(img)
            bic = (bic + 1) / 2
            bic = torch.clamp(bic, 0, 1)
            torchvision.utils.save_image(bic, res_path + f"bi{i:06}.png")
