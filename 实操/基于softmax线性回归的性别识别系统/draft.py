import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import torch
from torch import nn


device = torch.device("mps")
m = nn.Conv1d(16, 33, 3, stride=2).to(device)
Input = torch.randn(20, 16, 50).to(device)
output = m(Input)
print(Input.shape, output.shape)