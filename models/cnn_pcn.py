
from models.pcn_base import PCNetBase
from models.vanilla_pcn import PCLinear
from models.activations import *


def normalization(channels):
    return GroupNorm32(32, channels)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class PCConv2D(nn.Conv2d):
    def __init__(self, in_channels: int,  out_channels: int, activation: Activation,
                 kernel_size: int = 3, stride: int = 1, padding: int = 0, **kwargs):
        '''
        Projects to lower dim space with more channels, then reprojects to original space
        :param in_channels: initial # of channels in the incoming data. We will also reproject to this number
        :param channels: Channels of the latent space / initial projection space
        '''
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, **kwargs)
        self.activation = activation
        self.x = None
        self.activity = None
        self.grad = {'weights': None, 'bias': None}

    def forward(self, x: torch.Tensor):
        self.x = x

        # calculate the gradient of activity wrt kernel params
        self.activity = nn.Conv2d.forward(self, self.x)

        out = self.activation(self.activity)
        if self.bias is not None:
            out = out + self.bias[None, :, None, None]
        return out

    def backward(self, error: torch.Tensor):
        # fn_deriv = self.activation.deriv(self.activity)
        # # TODO: not sure if this is right
        # # out = nn.Conv2d.forward(self, error * fn_deriv)
        # out = error * (fn_deriv @ self.kernel_grad.T)
        # return out
        return -torch.autograd.grad(0.5 * torch.pow(error, 2).sum(), [self.x], retain_graph=True)[0]

    def update_gradient(self, error: torch.Tensor):
        # fn_deriv = self.activation.deriv(self.activity)
        # delta = error * (fn_deriv @ self.kernel_grad.T)
        # self.grad['weights'] = delta
        # if self.bias is not None:
        #     self.grad['bias'] = error.sum(0)
        delta = -torch.autograd.grad(0.5 * torch.pow(error, 2).sum(), [self.weight], retain_graph=True)[0]
        self.grad['weights'] = delta
        if self.bias is not None:
            self.grad['bias'] = -torch.autograd.grad(0.5 * torch.pow(error, 2).sum(), [self.bias], retain_graph=True)[0]


    def reset_grad(self):
        self.grad = {'weights': None, 'bias': None}


class PCConvNet(PCNetBase):
    def __init__(self, mu_dt: float):
        super().__init__(mu_dt)
        proj_in = PCConv2D(in_channels=1, out_channels=4, activation=RELU(), kernel_size=4, stride=2, padding=3)
        conv1 = PCConv2D(4, 8, RELU(), kernel_size=3, stride=2, padding=1)
        conv2 = PCConv2D(8, 16, RELU(), kernel_size=3, stride=2, padding=1)
        conv3 = PCConv2D(16, 32, RELU(), kernel_size=3, stride=2, padding=1)
        fc = PCLinear(32 * 4, 10, Identity())
        self.layers = nn.ModuleList([proj_in, conv1, conv2, conv3, fc])
        self.num_layers = len(self.layers)

        self.preds = [[] for _ in range(self.num_layers + 1)]
        self.errors = [[] for _ in range(self.num_layers + 1)]
        self.mus = [[] for _ in
                    range(self.num_layers + 1)]  # these are the outputs of each layer before the activations
