from LoadData import LinearDataset
from torch.utils.data import DataLoader
from model import LinerRegressionModel
from torch.optim import Adam
from torch.nn import MSELoss
import tqdm

trainDataset = LinearDataset()
validateDataset = LinearDataset()
batchSize = 512
numOfEpoch = 500
trainDataLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
validateDataLoader = DataLoader(validateDataset, batch_size=batchSize, shuffle=True)

model = LinerRegressionModel(7)
optim = Adam(model.parameters(), lr=0.1)
loss = MSELoss()


for Epoch in range(numOfEpoch):
    model.train()
    sumLoss = 0
    cnt = 0
    for X, Y in trainDataLoader:
        output = model(X)
        batchLoss = loss(output, Y)
        optim.zero_grad()
        batchLoss.backward()
        optim.step()
        sumLoss += batchLoss
        cnt += 1
    print(f"average loss:{sumLoss/cnt}")

model.eval()
sumLoss = 0
cnt = 0
for X, Y in validateDataLoader:
    output = model(X)
    sumLoss += loss(output, Y)
    cnt += 1

print(f"validation average loss:{sumLoss / cnt}")
