import torch
import numpy as np


def CCC_score(x, y):
    x_m = torch.mean(x)
    y_m = torch.mean(y)
    vx = x - x_m
    vy = y - y_m
    rho = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)))
    x_s = torch.std(x)
    y_s = torch.std(y)
    ccc = 2*rho*x_s*y_s/(x_s**2 + y_s**2 + (x_m - y_m)**2)
    return ccc


if __name__ == '__main__':
    y_true = torch.Tensor([3, -0.5, 2, 7])
    y_pred = torch.Tensor([2.5, 0.0, 2, 8])
    assert np.isclose(CCC_score(y_pred, y_true).item(), 0.97678916827853024, atol=1e-3)