import copy
import time

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
import numpy as np
from utils import weights_init


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(5, 4), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(5), requires_grad=False)
        self.f1 = nn.Linear(5, 16)
        self.f2 = nn.Linear(16, 2)
        self.apply(weights_init)

    def forward(self, x):
        x = F.linear(x, self.weight, self.bias)
        x = self.f2(self.f1(x))
        return x

    def unlearn(self):
        init.kaiming_normal_(self.weight)
        self.weight.requires_grad = True
        self.bias.requires_grad = True


model = Net()
x = np.array([[1, 1], [1, 1]])
y = np.array([[0., 0.], [1, 1]])
for i in range(100):
    torch.manual_seed(i)
    data = torch.Tensor(32, 10)
    labels = torch.Tensor(32, 10)
    loss_fn = torch.nn.MSELoss()
    loss1 = torch.sqrt(((data-labels)**2).sum())
    loss2 = torch.sqrt(loss_fn(data, labels)*data.numel())
    if loss1.item() != loss2.item():
        print('no')
exit()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.1)
for index in range(10):
    model.train()
    output = model(data)
    loss = loss_fn(output, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
model_weights = copy.deepcopy(model.state_dict())
model.unlearn()
for index in range(10):
    model.train()
    output = model(data)
    loss = loss_fn(output, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
unlearned_model_weights = copy.deepcopy(model.state_dict())
pass
