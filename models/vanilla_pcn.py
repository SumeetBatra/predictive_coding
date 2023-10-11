import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.autograd.functional as TAF

from typing import Callable


def linear_base_fn(activation_fn: Callable, input: torch.Tensor, weights: torch.Tensor, biases: torch.Tensor):
    return activation_fn(F.linear(input, weights, biases))


class RELU(nn.ReLU):
    def __init__(self):
        super().__init__()

    def deriv(self, x: torch.Tensor):
        '''
        Computes the derivative of relu at the point given by x
        '''
        out = self(x)
        out[out > 0] = 1.0
        return out


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

    def forward(self, x: torch.Tensor):
        self.x = x.clone()
        return self.activation(nn.Linear.forward(self, x))

    def backward(self, error: torch.Tensor):
        '''
        Computes the local gradient of the error of the output of this layer wrt layer params
        :param e: Error
        '''
        fn_deriv = self.activation.deriv(self.x @ self.weight)
        out = (error * fn_deriv) @ self.weight.T
        return out

    def update_gradient(self, error: torch.Tensor):
        fn_deriv = self.activation(self.x @ self.weight)
        delta = self.x.T @ (error * fn_deriv)
        self.grad['weights'] = delta
        if self.bias:
            self.grad['bias'] = error.sum(0)


class PCNet(nn.Module):
    def __init__(self, mu_dt):
        super().__init__()
        fc1 = PCLinear(784, 128, RELU())
        fc2 = PCLinear(128, 64, RELU())
        fc3 = PCLinear(64, 10, RELU())
        self.layers = nn.ModuleList([fc1, fc2, fc3])
        self.num_layers = len(self.layers)

        self.preds = [[] for _ in range(self.num_layers + 1)]
        self.errors = [[] for _ in range(self.num_layers + 1)]
        self.mus = [[] for _ in range(self.num_layers + 1)]
        self.mu_dt = mu_dt

    def set_input(self, x: torch.Tensor):
        # clamp the neural activity of 1st layer to the signal
        self.mus[0] = x.clone()

    def set_target(self, target: torch.Tensor):
        # clamp the activity of the last layer to be the target
        self.mus[-1] = target.clone()

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            for layer in self.layers:
                x = self.layer(x)  # nonlinearities happen within each layer
        return x

    def backprop_infer(self, x: torch.Tensor, labels: torch.Tensor, loss_fn: Callable):
        with torch.no_grad():
            device = next(self.parameters()).device
            out = self.forward(x)
            errors = [[] for i in range(self.num_layers + 1)]
            loss, dloss = TAF.vjp(lambda out: loss_fn(out, labels), out, torch.tensor(1).to(device))
            errors[-1] = dloss
            for i, layer in enumerate(reversed(self.layers)):
                error = layer.backward()
                errors[self.num_layers - i - 1] = error
