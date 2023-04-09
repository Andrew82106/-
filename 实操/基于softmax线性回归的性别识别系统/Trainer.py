import os
import tqdm
import torch
from torch import nn
import dirTool
import torchvision.transforms as transforms
import cv2 as cv
import config
from torch.utils.data import DataLoader, TensorDataset
from d2l import torch as d2l

maxlength = 784
batch_size = 256


def readRoute():  # 读取图片位置并且输出为列表
    root = config.configImageLoc()
    Training = []
    Validation = []

    training = [os.path.join(root, "Training", "female"), os.path.join(root, "Training", "male")]
    validation = [os.path.join(root, "Validation", "female"), os.path.join(root, "Validation", "male")]
    for i in dirTool.ToolBags.ls(training[0]):
        Training.append([os.path.join(training[0], i), 'female'])
    for i in dirTool.ToolBags.ls(training[1]):
        Training.append([os.path.join(training[1], i), 'male'])
    for i in dirTool.ToolBags.ls(validation[0]):
        Validation.append([os.path.join(validation[0], i), 'female'])
    for i in dirTool.ToolBags.ls(validation[1]):
        Validation.append([os.path.join(validation[1], i), 'male'])

    return Training, Validation


def loadDataset(RouteList):
    trans = transforms.ToTensor()
    img_list = []
    Type_list = []
    for i in tqdm.tqdm(RouteList):
        img = cv.imread(i[0])
        tensor = trans(img).flatten()
        tensor *= 10000
        for kk in range(50000):  # 数据集增强5w倍
            img_list.append(torch.squeeze(tensor.resize_(1, maxlength)))
            Type_list.append(0 if i[1] == 'female' else 1)
    imgs = torch.stack(img_list, dim=0)
    Type = torch.Tensor(Type_list)
    return DataLoader(dataset=TensorDataset(imgs, Type), batch_size=batch_size, shuffle=1)


# Dataset = loadDataset(readRoute()[0])
Dataset, Test = d2l.load_data_fashion_mnist(batch_size)
Dataset1 = Dataset


def init_weight(layer):
    if isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight, std=0.01)


def defModel():
    model = nn.Sequential(nn.Flatten(),
                          nn.Linear(maxlength, 10))  # 单层线性感知机
    model1 = nn.Sequential(
        nn.Flatten(),
        nn.Linear(maxlength, 2048),
        nn.ReLU(),
        nn.Linear(2048, 8192),
        nn.ReLU(),
        nn.Linear(8192, 10)
    )  # 含隐藏层的感知机
    model1.apply(init_weight)
    return model1


def defOptim(model):
    return torch.optim.SGD(params=model.parameters(), lr=0.1)


def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def train():
    model = defModel()
    Optimer = defOptim(model)
    loss = nn.CrossEntropyLoss(reduction='none')
    num_of_epoch = 200
    T = 0

    train_iter = Dataset
    model.train()
    for X, y in train_iter:
        if T > num_of_epoch:
            break
        y_hat = model(X)
        l = loss(y_hat, y.long())
        Optimer.zero_grad()
        l.mean().backward()
        Optimer.step()
        print(
            f'epoch{T + 1}:loss={1 - accuracy(y_hat, y) / len(y_hat)} numOfRight={accuracy(y_hat, y)} numOfAll={len(y_hat)}')
        """for i in model.parameters():
            print(i)"""
        T += 1
    return model


def predict(net):
    for x, y in Dataset1:
        print(x[0])
        print(y[0])
        print(net(x[0]))
        break


if __name__ == '__main__':
    # X = loadDataset(readRoute()[0])
    TrainedModel = train()
    # predict(TrainedModel)
    # 很明显，因为数据集太难了，模型跑不出来好效果，每次预测就和随机一样
    # 换了softmax用的那个数据集后模型马上就可以收敛了，6
