import os
import scipy.io as sio
import matplotlib.pyplot as plt
from argparse import ArgumentParser

args = ArgumentParser()
args.add_argument("--ker_path", type=str, default=None)
args.add_argument("--res_path", type=str, default=None)
args = args.parse_args()

kernels_path = args.ker_path
results_path = args.res_path

kernel_names = []
kernels = []
for filename in os.listdir(kernels_path):
    try:
        # if filename.startswith("hr2_0086"):
        #     print()
        mat = sio.loadmat(os.path.join(kernels_path, filename))
        mat = mat["Kernel"]
        kernel_names.append(filename[:filename.rfind(".")] + ".png")
        kernels.append(mat)
    except Exception:
        print(filename, end=" ")

for name, kernel in zip(kernel_names, kernels):
    plt.imsave(results_path + name, kernel)
