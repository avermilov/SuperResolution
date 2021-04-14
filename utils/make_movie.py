import PIL
import cv2
from argparse import ArgumentParser
import json
import numpy as np

import torch
from torchvision import transforms

from models.generators import newer_esrgan_generator
from models.generators.rdn import RDN


def torch_to_frame(img: torch.tensor) -> np.array:
    img = (img + 1) / 2
    img = torch.clamp(img, 0, 1)
    img = torch.squeeze(img, dim=0)
    img = img.cpu().detach().numpy()
    img = img.transpose(1, 2, 0)
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


parser = ArgumentParser()
parser.add_argument("json", type=str, default=None)
args = parser.parse_args()
json_path = args.json

with open(json_path, "r") as file:
    data = json.load(file)
    save_bicubic = data["save_bicubic"]
    save_name = data["save_name"]
    net_type = data["net_type"]
    movie_path = data["movie_path"]
    checkpoint_path = data["checkpoint_path"]
    start_frame = data["start_frame"]
    end_frame = data["end_frame"]
    scale = data["scale"]
    fps = data["fps"]

d = torch.load(checkpoint_path)

generator = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
net_type = net_type.lower()
if net_type == "rdn":
    generator = RDN(scale, 3, 64, 64, 16, 8).to(DEVICE)
elif net_type == "esrgen16":
    generator = newer_esrgan_generator.esrgan16(scale, pretrained=False).to(DEVICE)
elif net_type == "esrgen23":
    generator = newer_esrgan_generator.esrgan23(scale, pretrained=False).to(DEVICE)

generator.load_state_dict(d["generator"])
generator.eval()

vcsan = cv2.VideoCapture(movie_path)
suc, img = vcsan.read()
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
sh = (img.shape[1] * scale, img.shape[0] * scale)
out = cv2.VideoWriter(save_name + '.mp4', fourcc, fps, sh)
if save_bicubic:
    outbi = cv2.VideoWriter(save_name + 'bi.mp4', fourcc, fps, sh)
ind = 0
bic = transforms.Resize((sh[1], sh[0]), interpolation=PIL.Image.BICUBIC)
while suc:
    if start_frame <= ind <= end_frame:
        with torch.no_grad():
            # cv2.imshow("frame", img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255
            image = torch.from_numpy(img.transpose(2, 0, 1)).type(torch.FloatTensor)
            image = (image - 0.5) / 0.5
            image = torch.clamp(image, -1, 1)
            image = torch.unsqueeze(image, dim=0).to(DEVICE)
            image.requires_grad = False
            bicubic = transforms.Resize((sh[1], sh[0]), interpolation=PIL.Image.BICUBIC)(image)
            bicubic = torch_to_frame(bicubic)
            sr = generator(image)
            sr = torch_to_frame(sr)
            out.write(sr)
            if save_bicubic:
                bicubic = bic(image)
                bicubic = torch_to_frame(bicubic)
                outbi.write(bicubic)
            print(ind)
    ind += 1
    if ind > end_frame:
        break
    suc, img = vcsan.read()

vcsan.release()
cv2.destroyAllWindows()
