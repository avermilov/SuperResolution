import json
from argparse import ArgumentParser

import PIL
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from models.generators import rdn, newer_esrgan_generator, blur_rdn


def cut_image(image: torch.tensor, piece_count: int, padding: int) -> list:
    _, c, h, w = image.shape
    h //= piece_count
    w //= piece_count
    pieces = []
    for i in range(piece_count):
        for j in range(piece_count):
            pieces.append(image[:, :, i * h - teta(i) * padding:(i + 1) * h + teta(piece_count - 1 - i) * padding,
                          j * w - teta(j) * padding:(j + 1) * w + teta(piece_count - 1 - j) * padding])
    return pieces


def glue_image(pieces: list, piece_count: int, padding: int) -> torch.tensor:
    horiz = []
    for i in range(len(pieces)):
        pieces[i] = pieces[i][:, :, padding * 2 * teta(i // piece_count):
                                    pieces[i].shape[2] - padding * 2 * teta(piece_count - 1 - i // piece_count),
                    padding * 2 * teta(i % piece_count):
                    pieces[i].shape[3] - padding * 2 * teta(piece_count - 1 - i % piece_count)]
    for i in range(piece_count):
        horiz.append(torch.cat(pieces[i * piece_count:(i + 1) * piece_count], 3))
    image = torch.cat(horiz, 2)
    return image


def teta(x: int) -> int:
    if x != 0:
        return 1
    return 0


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
    net_type = data["net_type"].lower()
    save_prefix = data["save_prefix"]
    save_bicubic = data["save_bicubic"]
    DEVICE = data["device"]
    padding = data["padding"]
    cut = data["cut"]
    scale = data["scale"]

ds_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

net = None
if net_type == "rdn":
    net = rdn.RDN(2, 3, 64, 64, 16, 8).to(DEVICE)
elif net_type == "rdnblur":
    net = blur_rdn.BlurRDN(2, 3, 64, 64, 16, 8).to(DEVICE)
elif net_type == "esrgen16":
    net = newer_esrgan_generator.esrgan16(scale=2).to(DEVICE)
elif net_type == "esrgen23":
    net = newer_esrgan_generator.esrgan23(scale=2).to(DEVICE)

gen_dict = torch.load(net_path)["generator"]
net.load_state_dict(gen_dict)
net.eval()

ds = ImageFolder(img_path, transform=ds_transform)
loader = DataLoader(ds, batch_size=1, shuffle=False)

with torch.no_grad():
    for i, (img, _) in enumerate(loader):
        img = img.to(DEVICE)
        sr = img
        for _ in range(scale // 2):
            if cut != 1:
                pieces = cut_image(sr, cut, padding)
                sr_pieces = []
                for piece in pieces:
                    sr_pieces.append(net(piece))
                sr = glue_image(sr_pieces, cut, padding)
            else:
                sr = net(sr)
        sr = (sr + 1) / 2
        sr = torch.clamp(sr, 0, 1)
        torchvision.utils.save_image(sr, res_path + f"{save_prefix}_sr{i:06}.png")
        if save_bicubic:
            h, w = img.shape[-2:]
            bic = transforms.Resize((scale * h, scale * w), interpolation=PIL.Image.BICUBIC)(img)
            bic = (bic + 1) / 2
            bic = torch.clamp(bic, 0, 1)
            torchvision.utils.save_image(bic, res_path + f"{save_prefix}_bi{i:06}.png")
