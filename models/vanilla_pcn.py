import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pcn_base import PCNetBase
from models.layers import PCLinear
from typing import Callable


def linear_base_fn(activation_fn: Callable, input: torch.Tensor, weights: torch.Tensor, biases: torch.Tensor):
    return activation_fn(F.linear(input, weights, biases))


class PCNet(PCNetBase):
    def __init__(self, mu_dt: float, batch_size: int):
        super().__init__(mu_dt, batch_size)
        fc1 = PCLinear(784, 128, nn.ReLU())
        fc2 = PCLinear(128, 64, nn.ReLU())
        fc3 = PCLinear(64, 10, nn.Identity())
        self.layers = nn.ModuleList([fc1, fc2, fc3])
        self.num_layers = len(self.layers)


class GenerativePCNet(PCNet):
    def __init__(self, mu_dt: float, batch_size: int):
        super().__init__(mu_dt, batch_size)
        # generative model is a bit sensitive to choice of activation. Tanh does better than relu, probably b/c
        # mnist data is normalized to [0, 1]
        fc1 = PCLinear(10, 100, activation=nn.Tanh())
        fc2 = PCLinear(100, 300, activation=nn.Tanh())
        fc3 = PCLinear(300, 784, nn.Identity())
        self.layers = nn.ModuleList([fc1, fc2, fc3])
        self.num_layers = len(self.layers)

        self.preds = [[] for _ in range(self.num_layers + 1)]
        self.errors = [[] for _ in range(self.num_layers + 1)]
        self.mus = [[] for _ in
                    range(self.num_layers + 1)]  # these are the outputs of each layer before the activations
