import torch
import torch.nn as nn

from models.pcn_base import PCNetBase
from models.layers import PCLinear
from models.layers import PCConv2D


class PCConvNet(PCNetBase):
    def __init__(self, mu_dt: float, batch_size: int):
        super().__init__(mu_dt, batch_size)
        proj_in = PCConv2D(in_channels=1, out_channels=4, activation=nn.ReLU(), kernel_size=4, stride=2, padding=3)
        conv1 = PCConv2D(4, 8, nn.ReLU(), kernel_size=3, stride=2, padding=1)
        conv2 = PCConv2D(8, 16, nn.ReLU(), kernel_size=3, stride=2, padding=1)
        conv3 = PCConv2D(16, 32, nn.ReLU(), kernel_size=3, stride=2, padding=1)
        fc = PCLinear(32 * 4, 10, nn.Identity())
        self.layers = nn.ModuleList([proj_in, conv1, conv2, conv3, fc])
        self.num_layers = len(self.layers)

        self.preds = [[] for _ in range(self.num_layers + 1)]
        self.errors = [[] for _ in range(self.num_layers + 1)]
        self.mus = [[] for _ in
                    range(self.num_layers + 1)]  # these are the outputs of each layer before the activations
