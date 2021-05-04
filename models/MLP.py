import torch
from torch.autograd import Variable

import torch.nn as nn
class MLP(torch.nn.Module):
    def __init__(self, inputSize, outputSize,width=10):
        super(MLP, self).__init__()
        self.linear = torch.nn.Linear(inputSize, width)
        self.linear2=torch.nn.Linear(width,outputSize)

    def forward(self, x):
        #torch.nn.functional
        out = torch.nn.functional.relu(self.linear(x))
        out= self.linear2(out)
        return out