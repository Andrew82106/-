from baseConfig import *
from CNN.LoadData import *
from CNN.model import *
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import tqdm
from CNN.OptimLion import Lion


if __name__ == '__main__':
    batch_size = 10
    epoch = 50
    learning_rate = 0.001
    deviceName = 'mps'

    TrainData = PicDataset(archiveTrainingFemalePath, 0) + PicDataset(archiveTrainingMalePath, 1)
    TestData = PicDataset(archiveValidationFemalePath, 0) + PicDataset(archiveValidationMalePath, 1)
    TrainLoader = DataLoader(dataset=TrainData, batch_size=batch_size, shuffle=True)
    TestLoader = DataLoader(dataset=TestData, batch_size=batch_size, shuffle=True)

    device = torch.device(deviceName)
    model = CNN()
    model.to(device)
    # optimer = Adam(model.parameters(), lr=learning_rate)
    optimer = Lion(model.parameters(), lr=learning_rate)
    loss = CrossEntropyLoss()
    loss.to(device)

    for name, param in model.named_parameters():

        if 'weight' in name:
            nn.init.normal_(param, mean=0, std=0.01)
            print(name, param.shape, param.device)

        if 'bias' in name:
            nn.init.constant_(param, val=0)
            print(name, param.shape, param.device)

    for E in range(epoch):

        model.eval()
        with torch.no_grad():
            sumLoss = 0
            cnt = 0
            for img, label in TestLoader:
                img = img.to(device).type(torch.float32)
                label = label.to(device)
                predicted = model(img)
                sumLoss += loss(predicted, label)
                cnt += 1
            print(f"Test loss:{sumLoss / cnt}")


        sumLoss = 0
        cnt = 0
        model.train()
        for a, b in tqdm.tqdm(TrainLoader, desc=f'Epoch {E+1}'):
            a = a.to(device).type(torch.float32)
            b = b.to(device)
            res = model(a)
            batchLoss = loss(res, b)
            optimer.zero_grad()
            batchLoss.backward()
            optimer.step()
            sumLoss += batchLoss
            cnt += batch_size
        print(f'loss={float(sumLoss)/cnt}')
