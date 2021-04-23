import PIL
import cv2
from argparse import ArgumentParser
import json
import numpy as np
import pyprind
import skvideo.io as vio

import torch
from torchvision import transforms
from predict import inference_image
from models.generators import newer_esrgan_generator, rdn, blur_rdn


def torch_to_frame(img: torch.tensor) -> np.array:
    img = (img + 1) / 2
    img = torch.clamp(img, 0, 1)
    img = torch.squeeze(img, dim=0)
    img = img.cpu().detach().numpy()
    img = img.transpose(1, 2, 0)
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


if __name__ == "__main__":
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
        cut = data["cut"]
        padding = data["padding"]
        compression_level = data["compression_level"]

    generator = None
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    net_type = net_type.lower()
    if net_type == "rdn":
        generator = rdn.RDN(2, 3, 64, 64, 16, 8).to(DEVICE)
    elif net_type == "rdnblur":
        generator = blur_rdn.BlurRDN(2, 3, 64, 64, 16, 8).to(DEVICE)
    elif net_type == "esrgen16":
        generator = newer_esrgan_generator.esrgan16(2, pretrained=False).to(DEVICE)
    elif net_type == "esrgen23":
        generator = newer_esrgan_generator.esrgan23(2, pretrained=False).to(DEVICE)

    d = torch.load(checkpoint_path)
    generator.load_state_dict(d["generator"])
    generator.eval()

    vcsan = cv2.VideoCapture(movie_path)
    suc, img = vcsan.read()

    out = vio.FFmpegWriter(save_name + ".mp4", outputdict={
        '-vcodec': 'libx264',
        '-crf': str(compression_level),
        '-preset': 'veryslow'
    })
    out_bic = vio.FFmpegWriter(save_name + "bic.mp4", outputdict={
        '-vcodec': 'libx264',
        '-crf': str(compression_level),
        '-preset': 'veryslow'
    })

    ind = 0
    progbar = pyprind.ProgBar(end_frame - start_frame + 1, title="Inference")
    bic = transforms.Resize((img.shape[0] * scale, img.shape[1] * scale), interpolation=PIL.Image.BICUBIC)
    while suc:
        if start_frame <= ind <= end_frame:
            with torch.no_grad():
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255
                image = torch.from_numpy(img.transpose(2, 0, 1)).type(torch.FloatTensor)
                image = (image - 0.5) / 0.5
                image = torch.clamp(image, -1, 1)
                image = torch.unsqueeze(image, dim=0).to(DEVICE)
                image.requires_grad = False

                if save_bicubic:
                    bicubic_frame = torch_to_frame(bic(image))
                    out_bic.writeFrame(bicubic_frame)
                sr = inference_image(generator, image, scale, cut, padding)
                sr = torch_to_frame(sr)
                out.writeFrame(sr)
                progbar.update()
        ind += 1
        if ind > end_frame:
            break
        suc, img = vcsan.read()

    out.close()
    vcsan.release()
