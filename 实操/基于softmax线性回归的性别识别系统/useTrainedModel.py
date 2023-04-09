import torch
from torch import argmax
from torchvision import transforms

from main import CIFAR10_Model
from PIL import Image


def Predict(image_path: str):
    device = torch.device("mps")
    model = torch.load("./CIFAR10_Model.pth")
    # image_path = "/Users/andrewlee/Desktop/Projects/动手深度学习/实操/基于softmax线性回归的性别识别系统/archive/Validation/female/112944.jpg.jpg"
    img_ = Image.open(image_path).convert("RGB")
    resize = transforms.Resize((32, 32))
    ToTensor = transforms.ToTensor()
    img_ = resize(img_)
    img_ = ToTensor(img_)
    print(img_.size())
    img_ = img_.reshape(1, 3, 32, 32).to(device)

    res = model(img_).to(torch.device("cpu"))
    print(res)
    print(f"照片是{'女' if argmax(res, dim=-1).item() == 0 else '男'}性")


Predict("/Users/andrewlee/Desktop/Projects/动手深度学习/实操/基于softmax线性回归的性别识别系统/截屏2023-04-09 17.39.41.png")
Predict("/Users/andrewlee/Desktop/Projects/动手深度学习/实操/基于softmax线性回归的性别识别系统/截屏2023-04-09 17.27.57.png")
Predict("/Users/andrewlee/Desktop/Projects/动手深度学习/实操/基于softmax线性回归的性别识别系统/截屏2023-04-09 20.04.49.png")