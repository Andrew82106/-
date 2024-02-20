from torch.utils.data import Dataset
from makeData import generateData
import torch


class LinearDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.X, self.Y = generateData()

    def __getitem__(self, item):
        X_ = torch.tensor(self.X[item], dtype=torch.float32)
        Y_ = torch.tensor(self.Y[item], dtype=torch.float32)
        return X_, Y_

    def __len__(self):
        return len(self.X)


if __name__ == '__main__':
    X = LinearDataset()
    print("end")