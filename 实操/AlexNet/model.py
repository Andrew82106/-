import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from torch import nn
from torch.nn import Sequential, Conv2d, ReLU, MaxPool2d, Linear, Flatten, Dropout, Sigmoid, AvgPool2d


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.model = Sequential(
            Conv2d(1, 6, 5, padding=2),
            Sigmoid(),
            MaxPool2d(2, stride=2),
            Conv2d(6, 16, 5),
            AvgPool2d(2, stride=2),
            Flatten(),
            Linear(25*16, 120),
            Sigmoid(),
            Linear(120, 84),
            Sigmoid(),
            Linear(84, 10)
        )

    def forward(self, Input):
        return self.model(Input)


class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()
        self.partA = Sequential(
            Conv2d(3, 96, 11, stride=4),
            ReLU(),
            MaxPool2d(3, stride=2)
        )
        self.partB = Sequential(
            Conv2d(96, 256, 5, padding=2),
            ReLU(),
            MaxPool2d(3, stride=2)
        )
        self.partC = Sequential(
            Conv2d(256, 384, 3, padding=1),
            ReLU(),
            Conv2d(384, 384, 3, padding=1),
            ReLU(),
            Conv2d(384, 256, 3, padding=1),
            ReLU(),
            MaxPool2d(3, stride=2)
        )
        self.output = Sequential(
            Flatten(),
            Linear(5*5*256, 4096),
            ReLU(),
            Dropout(),
            Linear(4096, 4096),
            ReLU(),
            Dropout(),
            Linear(4096, 1000),
        )

    def forward(self, Input):
        partA_res = self.partA(Input)
        partB_res = self.partB(partA_res)
        partC_res = self.partC(partB_res)
        output = self.output(partC_res)
        return output


if __name__ == '__main__':
    device = torch.device("mps")
    print("USE Device: mps")
    x = torch.ones((1, 3, 224, 224), device=device)
    x1 = torch.ones((1, 1, 28, 28), device=device)
    net = AlexNet().to(device)
    net1 = LeNet().to(device)
    print(net)
    print(net1)
    res = net(x)
    res1 = net1(x1)
    print(res.shape, res)
    print(res1.shape, res1)