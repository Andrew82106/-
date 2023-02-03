import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim


def MakeData():  # 造数据函数！
    """
    X:[a, a, a]
    Y:[b]
    Y=Xw+b+eps
    :return:
    """
    X = torch.normal(0, 1, (200, 4))
    w = torch.tensor([1, 2, 3, 100], dtype=torch.float)
    b = -12.1
    Y = torch.matmul(X, w) + b
    Y += torch.normal(0, 0.0001, Y.shape)
    res = TensorDataset(X, Y)
    return DataLoader(dataset=res, batch_size=10, shuffle=1), X, Y


def Train(Data_iterator: DataLoader, X, Y):
    model = nn.Sequential(nn.Linear(4, 1))
    loss = nn.MSELoss()
    trainer = optim.SGD(model.parameters(), lr=0.1)
    epoch = 1
    while epoch < 100:
        for x, y in Data_iterator:
            l = loss(model(x).reshape(-1, 1), y.reshape(-1, 1))
            trainer.zero_grad()
            l.backward()
            trainer.step()

        print(f'epoch:{epoch} loss:{loss(model(X).reshape(-1, 1), Y.reshape(-1, 1))}')
        epoch += 1
    # 训练倒是大概写出来了，但是模型参数不知道是个啥情况


if __name__ == '__main__':
    Train(*MakeData())
