import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from torch import nn
from torch.nn import Sequential, Conv2d, ReLU, MaxPool2d, Linear, Flatten, Dropout, Sigmoid, AvgPool2d


class LeNet(nn.Module):

    def __init__(self, classify_types=1000):
        super(LeNet, self).__init__()
        self.model = Sequential(
            Conv2d(3, 6, 5, padding=2),
            Sigmoid(),
            MaxPool2d(2, stride=2),
            Conv2d(6, 16, 5),
            AvgPool2d(2, stride=2),
            Flatten(),
            Linear(25*16, 120),
            Sigmoid(),
            Linear(120, 84),
            Sigmoid(),
            Linear(84, classify_types)
        )

    def forward(self, Input):
        return self.model(Input)


class AlexNet(nn.Module):

    def __init__(self, classify_types=1000):
        super(AlexNet, self).__init__()
        self.partA = Sequential(
            Conv2d(3, 96, 11, stride=4),
            ReLU(inplace=True),  # 这里使用了inplace=True，应该是本地修改？
            MaxPool2d(3, stride=2)
        )
        self.partB = Sequential(
            Conv2d(96, 256, 5, padding=2),
            ReLU(inplace=True),
            MaxPool2d(3, stride=2)
        )
        self.partC = Sequential(
            Conv2d(256, 384, 3, padding=1),
            ReLU(inplace=True),
            Conv2d(384, 384, 3, padding=1),
            ReLU(inplace=True),
            Conv2d(384, 256, 3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(3, stride=2)
        )
        self.output = Sequential(
            Flatten(),
            Linear(5*5*256, 4096),
            ReLU(inplace=True),
            Dropout(),
            Linear(4096, 4096),
            ReLU(inplace=True),
            Dropout(),
            Linear(4096, classify_types),
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