import torch
import torch.nn as nn

from models.layers import PCLinear, PCConv2D, PCSequential
from models.pcn_base import PCNetBase


class PCDiffusion(PCNetBase):
    def __init__(self, mu_dt: float, batch_size: int, n_diffusion_steps: int = 10):
        super().__init__(mu_dt, batch_size)

        layers = []
        # for _ in range(n_diffusion_steps - 1):
        #     fc = PCLinear(784, 784, nn.ReLU())
        #     layers.append(fc)
        layers = [PCLinear(784, 256, nn.ReLU()),
                  PCLinear(256, 64, nn.ReLU()),
                  PCLinear(64, 32, nn.ReLU()),
                  PCLinear(32, 64, nn.ReLU()),
                  PCLinear(64, 128, nn.ReLU()),
                  PCLinear(128, 256, nn.ReLU()),
                  PCLinear(256, 784, nn.Identity())]

        self.layers = nn.ModuleList(layers)
        self.num_layers = len(self.layers)

        self.preds = [[] for _ in range(self.num_layers + 1)]
        self.errors = [[] for _ in range(self.num_layers + 1)]
        self.mus = [[] for _ in
                    range(self.num_layers + 1)]  # these are the outputs of each layer before the activations
