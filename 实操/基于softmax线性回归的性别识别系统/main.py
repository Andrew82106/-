import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import dirTool
import os
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, CrossEntropyLoss
from torch.optim import SGD
from torchvision import transforms


class ImageDataset(Dataset):

    def __init__(self, rootLoc_, label, pic_size, Training=False):
        self.rootLoc = rootLoc_
        self.ImageLoc = os.path.join(rootLoc_, "archive", "Validation" if not Training else "Training", label)
        self.ImageList = dirTool.ToolBags.ls(self.ImageLoc)
        self.label = label
        self.resize = transforms.Resize(pic_size)
        self.ToTensor = transforms.ToTensor()

    def __getitem__(self, index):
        image_name = self.ImageList[index]
        image_path = os.path.join(self.ImageLoc, image_name)
        img_ = Image.open(image_path)
        img_ = self.resize(img_)
        img_ = self.ToTensor(img_)
        label_ = 0 if self.label == 'female' else 1
        return img_, label_

    def __len__(self):
        return len(self.ImageList)


class CIFAR10_Model(nn.Module):
    def __init__(self):
        super(CIFAR10_Model, self).__init__()
        self.model = nn.Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 2)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    rootLoc = "/Users/andrewlee/Desktop/Projects/动手深度学习/实操/基于softmax线性回归的性别识别系统"
    device = torch.device("mps")

    TrainDataset_female = ImageDataset(rootLoc, 'female', (32, 32), True)
    TrainDataset_male = ImageDataset(rootLoc, 'male', (32, 32), True)
    TrainDataset = (TrainDataset_male + TrainDataset_female)
    TrainDataset = DataLoader(TrainDataset, batch_size=128, shuffle=True)

    ValidateDataset_female = ImageDataset(rootLoc, 'female', (32, 32), False)
    ValidateDataset_male = ImageDataset(rootLoc, 'male', (32, 32), False)
    ValidateDataset = ValidateDataset_male + ValidateDataset_female
    ValidateDataset = DataLoader(ValidateDataset, batch_size=32, shuffle=True)

    model = CIFAR10_Model().to(device)
    optimer = SGD(model.parameters(), lr=0.01)
    lossFun = CrossEntropyLoss().to(device)

    num_of_epoch = 20
    cnt = 0
    for epoch in range(num_of_epoch):

        print(f"---------epoch{epoch}----------")
        loss_sum = 0
        for img, label in TrainDataset:
            img = img.to(device)
            label = label.to(device)
            output = model(img)
            loss = lossFun(output, label)
            loss_sum += loss
            optimer.zero_grad()
            loss.backward()
            optimer.step()
            cnt += 1
            if cnt % 40 == 0:
                print(f"loss:{loss}")
        print(f"epoch:{epoch} loss:{loss_sum}")

        total_test_loss = 0
        with torch.no_grad():

            for img, label in ValidateDataset:
                img = img.to(device)
                label = label.to(device)
                output = model(img)
                loss = lossFun(output, label)
                total_test_loss += loss.item()
        print(f"loss in Validate dataset: {total_test_loss}")

    torch.save(model, "./CIFAR10_Model.pth")

    print("end")
