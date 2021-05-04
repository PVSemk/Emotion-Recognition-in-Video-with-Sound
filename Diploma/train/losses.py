import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



class CCCLoss(nn.Module):
    def __init__(self, digitize_num, range=(-1, 1)):
        super(CCCLoss, self).__init__()
        self.digitize_num = digitize_num
        self.range = range
        if self.digitize_num:
            bins = np.linspace(*self.range, num=self.digitize_num)
            self.bins = torch.as_tensor(bins, dtype=torch.float32).view((1, -1))

    def forward(self, x, y):
        # the target y is continuous value (BS, )
        # the input x is either continuous value (BS, ) or probability output(digitized)
        y = y.view(-1)
        if torch.all(y == y[0]):
            y[0] = y[0] + 1e-5
        if self.digitize_num != 1:
            x = F.softmax(x, dim=-1)
            x = (self.bins.type_as(x) * x).sum(-1)  # expectation

        x = x.view(-1)
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        rho = torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.pow(vx, 2))) * torch.sqrt(torch.sum(torch.pow(vy, 2))))
        x_m = torch.mean(x)
        y_m = torch.mean(y)
        x_s = torch.std(x)
        y_s = torch.std(y)
        ccc = 2*rho*x_s*y_s/(torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2))
        return 1-ccc


class CrossEntropyLoss(nn.Module):
    def __init__(self, digitize_num, range=(-1, 1)):
        super(CrossEntropyLoss, self).__init__()
        self.digitize_num = digitize_num
        if self.digitize_num:
            self.range = range
            self.edges = torch.linspace(*self.range, steps=self.digitize_num+1)
    def forward(self, x, y):
        # the target y is continuous value (BS, )
        # the input x is  probability output(digitized)
        y = y.view(-1)
        y_dig = torch.bucketize(y, self.edges.to(y.device), right=True) - 1
        y_dig[y_dig == self.digitize_num] = self.digitize_num - 1
        y = y_dig
        return F.cross_entropy(x, y)