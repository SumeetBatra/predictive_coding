
from models.pcn_base import PCNetBase
from models.activations import *
from typing import Callable


def linear_base_fn(activation_fn: Callable, input: torch.Tensor, weights: torch.Tensor, biases: torch.Tensor):
    return activation_fn(F.linear(input, weights, biases))


class PCLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, activation: nn.Module, **kwargs):
        '''
        A generic subclass for any type of torch layer. When running inference, we use this class to calculate the
        local gradients wrt the loss on the predicted outgoing signal
        '''
        super().__init__(in_features, out_features, **kwargs)
        self.activation = activation
        self.x = None
        self.grad = {'weights': None, 'bias': None}
        nn.init.normal_(self.weight, 0., 0.05)
        self.bias.data = torch.zeros_like(self.bias.data)

    def forward(self, x: torch.Tensor):
        bs = x.shape[0]
        x = x.view(bs, -1)
        self.x = x
        # TODO: not sure why the bias term comes after the activation. Pretty sure this is wrong
        out = self.activation(self.x @ self.weight.T)
        if self.bias is not None:
            out = out + self.bias
        return out

    def backward(self, error: torch.Tensor):
        '''
        Computes the local gradient of the error of the output of this layer wrt layer params
        :param error: Error
        '''
        # fn_deriv = self.activation.deriv(self.x @ self.weight.T)
        # out = (error * fn_deriv) @ self.weight
        # return out
        return -torch.autograd.grad(0.5 * torch.pow(error, 2).sum(), [self.x], retain_graph=True)[0]

    def update_gradient(self, error: torch.Tensor):
        fn_deriv = self.activation.deriv(self.x @ self.weight.T)
        delta = self.x.T @ (error * fn_deriv)
        self.grad['weights'] = delta.T
        if self.bias is not None:
            self.grad['bias'] = error.sum(0)

        # delta = -torch.autograd.grad(0.5 * torch.pow(error, 2).sum(), [self.weight], retain_graph=True)[0]
        # self.grad['weights'] = delta
        # if self.bias is not None:
        #     self.grad['bias'] = -torch.autograd.grad(0.5 * torch.pow(error, 2).sum(), [self.bias], retain_graph=True)[0]

    def reset_grad(self):
        self.grad = {'weights': None, 'bias': None}


class PCNet(PCNetBase):
    def __init__(self, mu_dt):
        super().__init__(mu_dt)
        fc1 = PCLinear(784, 128, RELU())
        fc2 = PCLinear(128, 64, RELU())
        fc3 = PCLinear(64, 10, Identity())
        self.layers = nn.ModuleList([fc1, fc2, fc3])
        self.num_layers = len(self.layers)


class GenerativePCNet(PCNet):
    def __init__(self, mu_dt: float):
        super().__init__(mu_dt)
        # generative model is a bit sensitive to choice of activation. Tanh does better than relu, probably b/c
        # mnist data is normalized to [0, 1]
        fc1 = PCLinear(10, 100, activation=TANH())
        fc2 = PCLinear(100, 300, activation=TANH())
        fc3 = PCLinear(300, 784, Identity())
        self.layers = nn.ModuleList([fc1, fc2, fc3])
        self.num_layers = len(self.layers)

        self.preds = [[] for _ in range(self.num_layers + 1)]
        self.errors = [[] for _ in range(self.num_layers + 1)]
        self.mus = [[] for _ in
                    range(self.num_layers + 1)]  # these are the outputs of each layer before the activations
