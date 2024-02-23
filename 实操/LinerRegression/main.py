"""
代码中loss过大的问题在于LinearDataset给出的Y的形状不对！
"""

from LoadData import LinearDataset
from torch.utils.data import DataLoader
from model import LinerRegressionModel
from torch.optim import Adam, SGD
from OptimLion import Lion
from torch.nn import MSELoss, L1Loss
from Utils import plot_paraLst, plot_muti_Line
from makeData import numOfX

trainDataset = LinearDataset()
validateDataset = LinearDataset()
batchSize = 32
numOfEpoch = 100
trainDataLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
validateDataLoader = DataLoader(validateDataset, batch_size=batchSize, shuffle=True)

model = LinerRegressionModel(numOfX)

for para in model.parameters():
    para.data.normal_(0, 0.1)

optim = SGD(model.parameters(), lr=0.000001)
loss = L1Loss()

paraLst = [[] for _ in range(numOfX)]
biasLst = []
lossLst = []

for Epoch in range(numOfEpoch):
    model.train()
    sumLoss = 0
    cnt = 0
    for X, Y in trainDataLoader:
        # print("Start Batch")
        for name, param in model.named_parameters():
            if 'bias' in name:
                biasLst.append(float(param.data[0]))
                # print(f"bias:{float(param.data[0])}")
                continue
            for Index in range(numOfX):
                paraLst[Index].append(float(param.data[0][Index]))
                # print(f"para{Index}: {float(param.data[0][Index])}")

        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"Layer: {name}, Weight: {float(param.data)}, Gradient: {float(param.grad)}")

        output = model(X)
        batchLoss = loss(output, Y)
        optim.zero_grad()
        batchLoss.backward()
        optim.step()
        sumLoss += batchLoss
        lossLst.append(float(batchLoss))
        cnt += 1
    print(f"Epoch{Epoch} average loss:{sumLoss / cnt}")

model.eval()
sumLoss = 0
cnt = 0
for X, Y in validateDataLoader:
    output = model(X)
    sumLoss += loss(output, Y)
    cnt += 1

print(f"validation average loss:{sumLoss / cnt}")

plot_paraLst(paraLst)
plot_muti_Line({"loss history": lossLst})
plot_muti_Line({"bias history": biasLst})

for name, param in model.named_parameters():
    print(name, param)
