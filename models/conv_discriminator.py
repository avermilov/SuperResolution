import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvDiscriminator(nn.Module):
    def __init__(self, num_channels: int, num_features: int, num_deep_layers: int):
        super(ConvDiscriminator, self).__init__()
        self.in_channels = num_channels
        self.num_filters = num_features
        self.stride = 2
        self.kernel_size = 3
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.num_filters,
                               kernel_size=self.kernel_size, padding=1,
                               stride=self.stride)
        self.deep_layers = nn.ModuleList([nn.Conv2d(in_channels=self.num_filters,
                                                    out_channels=self.num_filters,
                                                    kernel_size=self.kernel_size, padding=1,
                                                    stride=self.stride)
                                          for _ in range(num_deep_layers)])
        self.conv4 = nn.Conv2d(in_channels=self.num_filters,
                               out_channels=1,
                               kernel_size=self.kernel_size, padding=1,
                               stride=self.stride)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = F.relu(self.conv1(x))
        for conv_layer in self.deep_layers:
            x = F.relu(conv_layer(x))
        x = self.conv4(x)
        return x

    def requires_grad(self, requires_grad: bool):
        if requires_grad:
            self.train()
            for param in self.parameters():
                param.requires_grad = True
        else:
            self.eval()
            for param in self.parameters():
                param.requires_grad = False
