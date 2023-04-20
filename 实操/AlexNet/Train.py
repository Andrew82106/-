from model import AlexNet
from model import LeNet

import os

import torch
from torch import nn
from torch.optim import Adam
from OptimLion import Lion
from LoadData import Generate_dataLoader
from torch.nn import CrossEntropyLoss


device_name = 'mps'
device = torch.device(device_name)
print(f"Info: use device:{device_name}")
trainLoader, testLoader = Generate_dataLoader(batch_size=64, pic_size=[28, 28])
AlexNetModel = AlexNet(120).to(device)
LeNet = LeNet(120).to(device)

for name, param in AlexNetModel.named_parameters():

    if 'weight' in name:
        nn.init.normal_(param, mean=0, std=0.01)

    if 'bias' in name:
        nn.init.constant_(param, val=0)


for name, param in LeNet.named_parameters():

    if 'weight' in name:
        nn.init.normal_(param, mean=0, std=0.01)

    if 'bias' in name:
        nn.init.constant_(param, val=0)


# optimer = Adam(AlexNetModel.parameters(), lr=0.01)
optimer = Lion(LeNet.parameters())
Loss = CrossEntropyLoss().to(device)

Epoch = 15

for epoch in range(Epoch):
    print(f"____________epoch{epoch+1}____________")

    AlexNetModel.train()
    loss_sum = 0
    cnt = 1
    for img_, label_ in trainLoader:
        img_ = img_.to(device)
        img_ = img_.type(torch.float32)  # 是要转格式才能用的
        label_ = label_.to(device)
        # label_ = label_.type(torch.float32)  # 是要转格式才能用的
        # output = AlexNetModel(img_)
        output = LeNet(img_)
        loss = Loss(output, label_)
        optimer.zero_grad()  # 一定记得梯度清零
        loss.backward()
        optimer.step()
        if cnt % 100 == 0:
            print(f"Training:average loss:{loss/cnt}")
        cnt += 1
        loss_sum += loss
    print(f"Info: epoch={epoch + 1} average loss={loss_sum/cnt}")

    AlexNetModel.eval()
    with torch.no_grad():

        total = 0
        correct = 0
        for img_, label_ in testLoader:
            img_ = img_.to(device)
            img_ = img_.type(torch.float32)
            label_ = label_.to(device)
            # output = AlexNetModel(img_)
            output = LeNet(img_)
            _, predicted = torch.max(output.data, 1)
            # 计算正确预测的数量
            total += label_.size(0)
            correct += (predicted == label_).sum().item()
        print(f"Info: test accuracy={100 * correct / total}%")
