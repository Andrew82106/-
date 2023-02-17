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
    numOfItem = 20000  # 样本数
    w = torch.tensor([10, 20.2, 23, 3.21], dtype=torch.float)  # 参数矩阵
    X = torch.normal(0, 1, (numOfItem, len(w)))
    b = -12.1
    Y = torch.matmul(X, w) + b  # 计算Y
    Y += torch.normal(0, 0.0001, Y.shape)  # 加上随机扰动
    res = TensorDataset(X, Y)  # 打包为一个dataset
    return DataLoader(dataset=res, batch_size=20, shuffle=1), X, Y


def Train(Data_iterator: DataLoader, X, Y):
    model = nn.Sequential(nn.Linear(4, 1))
    loss = nn.MSELoss()
    trainer = optim.SGD(model.parameters(), lr=0.001)
    epoch = 1
    while epoch < 30:
        for x, y in Data_iterator:
            l = loss(model(x).reshape(-1, 1), y.reshape(-1, 1))
            trainer.zero_grad()
            l.backward()
            trainer.step()

        print(f'epoch:{epoch} loss:{loss(model(X).reshape(-1, 1), Y.reshape(-1, 1))}')
        epoch += 1
    for x in model.parameters():
        print(x)
        # 可以通过这种方式访问模型参数
        # 因为model.parameters()返回的是一个迭代器，因此只能这样迭代才能看明白
    return model, loss


if __name__ == '__main__':
    Model, loss = Train(*MakeData())
    Z, X_, Y_ = MakeData()  # 重新造一个测试集出来
    print(f"测试集loss为{loss(Model(X_).reshape(-1, 1), Y_.reshape(-1, 1))}")
