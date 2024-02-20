import torch.nn as nn
from torch.nn import Linear


class LinerRegressionModel(nn.Module):
    def __init__(self, numOfInput):
        super().__init__()
        self.Linear = Linear(numOfInput, numOfInput*2)
        self.Linear1 = Linear(numOfInput*2, 1)

    def forward(self, data):
        return self.Linear1(self.Linear(data))