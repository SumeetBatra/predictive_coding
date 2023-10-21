import torch
import torch.nn as nn
import torch.nn.functional as F


class Activation(nn.Module):
    def forward(self, x: torch.Tensor):
        raise NotImplementedError

    def deriv(self, x: torch.Tensor):
        raise NotImplementedError


class RELU(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return F.relu(x, inplace=False)

    def deriv(self, x: torch.Tensor):
        '''
        Computes the derivative of relu at the point given by x
        '''
        out = F.relu(x)
        out[out > 0] = 1.0
        return out


class TANH(Activation):
    def forward(self, inp):
        return torch.tanh(inp)

    def deriv(self, inp):
        return 1.0 - torch.tanh(inp) ** 2.0


class Identity(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x

    def deriv(self, x: torch.Tensor):
        return torch.ones((1,)).to(x.device)