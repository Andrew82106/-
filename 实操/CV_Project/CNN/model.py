import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 512),
            Linear(512, 64),
            Linear(64, 2)
        )

    def forward(self, x):
        return self.model(x)