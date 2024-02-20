import os.path
import torch

from baseConfig import *
from CNN.LoadData import *
from CNN.model import *
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss, MSELoss
import tqdm
from CNN.OptimLion import Lion
from CNN.drawFig import *


def loadSavedModel(modelStructure, modelPath):
    checkpoint = torch.load(modelPath)
    modelStructure.load_state_dict(checkpoint)
    return modelStructure


def saveModel(TrainedModel, modelPath):
    torch.save(TrainedModel.state_dict(), modelPath)



if __name__ == '__main__':
    batch_size = 256
    epoch = 100
    learning_rate = 0.01
    deviceName = 'mps'

    TrainData = PicDataset(archiveTrainingFemalePath, 0) + PicDataset(archiveTrainingMalePath, 1)
    TestData = PicDataset(archiveValidationFemalePath, 0) + PicDataset(archiveValidationMalePath, 1)
    TrainLoader = DataLoader(dataset=TrainData, batch_size=batch_size, shuffle=True)
    TestLoader = DataLoader(dataset=TestData, batch_size=batch_size, shuffle=True)

    device = torch.device(deviceName)
    model = CNN()
    model.to(device)
    # optimer = Adam(model.parameters(), lr=learning_rate)
    # optimer = SGD(model.parameters(), lr=learning_rate)
    optimer = Lion(model.parameters(), lr=learning_rate)
    loss = CrossEntropyLoss()
    loss1 = MSELoss()
    loss.to(device)
    loss1.to(device)

    for name, param in model.named_parameters():

        if 'weight' in name:
            nn.init.normal_(param, mean=0, std=0.01)
            print(name, param.shape, param.device)


        if 'bias' in name:
            nn.init.constant_(param, val=0)
            print(name, param.shape, param.device)

    acc = -1
    bestModel = None
    acc_history = []
    loss_history = []
    vali_loss_history = []
    for E in range(epoch):

        sumLoss = 0
        cnt = 0
        model.train()
        for a, b in tqdm.tqdm(TrainLoader, desc=f'Epoch {E+1}'):
            a = a.to(device).type(torch.float32)
            b = b.to(device)
            res = model(a)
            batchLoss = loss(res, b)
            cnt += 1
            optimer.zero_grad()
            batchLoss.backward()
            optimer.step()
            sumLoss += batchLoss
        print(f'loss={float(sumLoss)/cnt}')
        loss_history.append(float(sumLoss)/cnt)

        model.eval()
        sumCount = 0
        allCount = 0
        sumLoss = 0
        cnt = 0
        with torch.no_grad():
            for img, label in TestLoader:
                img = img.to(device).type(torch.float32)
                label = label.to(device)
                P = model(img)
                predicted = torch.argmax(P, dim=1)
                batchLoss = loss(P, label)
                sumLoss += batchLoss
                cnt += 1
                equal_elements = torch.eq(predicted, label)
                count = torch.sum(equal_elements).item()
                sumCount += count
                allCount += len(predicted)
            print(f"Test acc:{sumCount / allCount}")
            print(f"Test loss:{float(sumLoss) / cnt}")
            if sumCount / allCount > acc:
                acc = sumCount / allCount
                bestModel = model
                print("got best performance")
            acc_history.append(sumCount / allCount)
            vali_loss_history.append(float(sumLoss) / cnt)

    saveModel(bestModel, os.path.join(CNNPath, 'modelCache', f"CNN_best_acc={acc}.pth"))
    draw_line_chart(acc_history, X='epoch', Y='accuracy', title='accuracy history', saveLocation=os.path.join(CNNPath, 'trainingLogs', 'accuracy.png'))
    draw_line_chart(loss_history, X='epoch', Y='loss', title='loss history', saveLocation=os.path.join(CNNPath, 'trainingLogs', 'loss.png'))
    draw_line_chart(vali_loss_history, X='epoch', Y='loss', title='validation loss history', saveLocation=os.path.join(CNNPath, 'trainingLogs', 'vali_loss.png'))