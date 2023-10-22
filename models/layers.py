import torch
import torch.nn as nn


class PCLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, activation: nn.Module, **kwargs):
        '''
        A generic subclass for any type of torch layer. When running inference, we use this class to calculate the
        local gradients wrt the loss on the predicted outgoing signal
        '''
        super().__init__(in_features, out_features, **kwargs)
        self.activation = activation
        self.x = None
        self.x_shape = None
        self.grad = {'weights': None, 'bias': None}
        nn.init.normal_(self.weight, 0., 0.05)
        self.bias.data = torch.zeros_like(self.bias.data)

    def forward(self, x: torch.Tensor):
        bs = x.shape[0]
        self.x_shape = x.shape
        x = x.view(bs, -1)
        self.x = x
        out = self.activation(nn.Linear.forward(self, self.x))
        return out

    def reset_grad(self):
        self.grad = {'weights': None, 'bias': None}


def normalization(channels):
    return GroupNorm32(32, channels)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class PCConv2D(nn.Conv2d):
    def __init__(self, in_channels: int,  out_channels: int, activation: nn.Module,
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
        # if self.bias is not None:
        #     out = out + self.bias[None, :, None, None]
        return out

    def reset_grad(self):
        self.grad = {'weights': None, 'bias': None}


class PCSequential(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)
        self.mu = None
        self.grad = {'weights': None, 'bias': None}

    def forward(self, x: torch.Tensor):
        self.mu = x
        return nn.Sequential.forward(x)

    def reset_grad(self):
        self.grad = {'weights': None, 'bias': None}
