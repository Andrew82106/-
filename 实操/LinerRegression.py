import torch
from torch.utils.data import DataLoader, TensorDataset


def MakeData() -> DataLoader:
    """
    X:[a, a, a]
    Y:[b]
    Y=Xw+b+eps
    :return:
    """
    X = torch.normal(0, 1, (200, 3))
    w = torch.tensor([1, 2, 3], dtype=torch.float)
    b = -12.1
    Y = torch.matmul(X, w) + b
    Y += torch.normal(0, 0.0001, Y.shape)
    res = TensorDataset(X, Y)
    return DataLoader(dataset=res, batch_size=3, shuffle=1)


if __name__ == '__main__':
    Data_iterator = MakeData()  # 造数据函数！
    for x, y in Data_iterator:
        print(x[0])
        print(y[0])
        break