from baseConfig import *
from CNN.LoadData import *
from CNN.model import *


if __name__ == '__main__':
    batch_size = 10

    TrainData = PicDataset(archiveTrainingFemalePath, 0) + PicDataset(archiveTrainingMalePath, 1)
    TestData = PicDataset(archiveValidationFemalePath, 0) + PicDataset(archiveValidationMalePath, 1)
    TrainLoader = DataLoader(dataset=TrainData, batch_size=batch_size, shuffle=True)
    TestLoader = DataLoader(dataset=TestData, batch_size=batch_size, shuffle=True)

    model = CNN()
    for a, b in TestLoader:
        res = model(a)
        print(res)