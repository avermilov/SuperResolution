import PIL
from torch import nn
from torchvision import transforms


class SRCNN(nn.Module):
    def __init__(self, num_channels=3, scale: int = 2):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)
        self.scale = scale

    def forward(self, x):
        x = transforms.Resize(x.shape[-1] * self.scale,
                              interpolation=PIL.Image.BICUBIC)(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
