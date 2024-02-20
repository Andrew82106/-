from torch.utils.data import Dataset
from makeData import generateData


class LinearDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.X, self.Y = generateData()

    def __getitem__(self, item):
        X_ = self.X[item]
        Y_ = self.Y[item]
        return X_, Y_

    def __len__(self):
        return len(self.X)


if __name__ == '__main__':
    X = LinearDataset()
    print("end")