import os
from argparse import ArgumentParser

import cv2


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


argparser = ArgumentParser()
argparser.add_argument("img_path", type=str, default=None)
argparser.add_argument("save_path", type=str, default=None)
argparser.add_argument("save_name", type=str, default=None)
args = argparser.parse_args()

img_path = args.img_path
res_path = args.save_path
save_prefix = args.save_name

img = load_images_from_folder(img_path)

height, width, _ = img[0].shape

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(res_path + save_prefix + '.avi', fourcc, 2.0, (width, height))

for image in img:
    video.write(image)

cv2.destroyAllWindows()
video.release()
