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

class TANH(nn.Tanh):
    def forward(self, inp):
        return torch.tanh(inp)

    def deriv(self, inp):
        return 1.0 - torch.tanh(inp) ** 2.0


class Identity(nn.Identity):
    def __init__(self):
        super().__init__()

    def deriv(self, x: torch.Tensor):
        return torch.ones((1,)).to(x.device)


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
        self.x = x.clone()
        # TODO: not sure why the bias term comes after the activation. Pretty sure this is wrong
        out = self.activation(self.x @ self.weight.T)
        if self.bias is not None:
            out += self.bias
        return out

    def backward(self, error: torch.Tensor):
        '''
        Computes the local gradient of the error of the output of this layer wrt layer params
        :param e: Error
        '''
        fn_deriv = self.activation.deriv(self.x @ self.weight.T)
        out = (error * fn_deriv) @ self.weight
        return out

    def update_gradient(self, error: torch.Tensor):
        fn_deriv = self.activation.deriv(self.x @ self.weight.T)
        delta = self.x.T @ (error * fn_deriv)
        self.grad['weights'] = delta.T
        if self.bias is not None:
            self.grad['bias'] = error.sum(0)

    def reset_grad(self):
        self.grad = {'weights': None, 'bias': None}


class PCNet(nn.Module):
    def __init__(self, mu_dt):
        super().__init__()
        fc1 = PCLinear(784, 128, RELU())
        fc2 = PCLinear(128, 64, RELU())
        fc3 = PCLinear(64, 10, Identity())
        self.layers = nn.ModuleList([fc1, fc2, fc3])
        self.num_layers = len(self.layers)

        self.preds = [[] for _ in range(self.num_layers + 1)]
        self.errors = [[] for _ in range(self.num_layers + 1)]
        self.mus = [[] for _ in range(self.num_layers + 1)]  # these are the outputs of each layer before the activations
        self.mu_dt = mu_dt

    @property
    def params(self):
        return self.layers

    def set_input(self, x: torch.Tensor):
        # clamp the neural activity of 1st layer to the signal
        self.mus[0] = x.clone()

    def set_target(self, target: torch.Tensor):
        # clamp the activity of the last layer to be the target
        self.mus[-1] = target.clone()

    def update_grads(self):
        for l in range(self.num_layers):
            self.layers[l].update_gradient(self.errors[l+1])

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            for layer in self.layers:
                x = layer(x)  # nonlinearities happen within each layer
        return x

    def reset(self):
        self.preds = [[] for _ in range(self.num_layers + 1)]
        self.errors = [[] for _ in range(self.num_layers + 1)]
        self.mus = [[] for _ in range(self.num_layers + 1)]

    def reset_mus(self, batch_size: int, init_std: float):
        device = next(self.parameters()).device
        for l in range(self.num_layers):
            self.mus[l] = torch.empty(batch_size, self.layers[l].in_features).normal_(mean=0, std=init_std).to(device)

    def propagate_mu(self):
        for l in range(1, self.num_layers):
            self.mus[l] = self.layers[l-1].forward(self.mus[l-1])

    def inference(self, n_iters: int, fixed_preds: bool = False, predict_label_inputs: bool = False):
        for n in range(1, self.num_layers + 1):
            # TODO: I think this is already computed when we call propagate_mu ?
            self.preds[n] = self.layers[n-1].forward(self.mus[n-1])
            self.errors[n] = self.mus[n] - self.preds[n]

        for _ in range(n_iters):
            if predict_label_inputs:
                # when predict_label_inputs = False, mus[0] is clamped to be the input so there is no delta.
                # however when predict_label_inputs = True, only mus[-1] is clamped to be the target, so we need to
                # compute a delta for mus[0] and update mus[0] as well
                delta = self.layers[0].backward(self.errors[1])
                self.mus[0] = self.mus[0] + self.mu_dt * delta
            # step the activities to minimize free energy
            for l in range(1, self.num_layers):
                # TODO: why is delta computed this way?
                delta = self.layers[l].backward(self.errors[l + 1]) - self.errors[l]
                # TODO: so is mu_d the learning rate for the inference phase?
                self.mus[l] = self.mus[l] + self.mu_dt * delta

            # recompute error in predictions
            for n in range(1, self.num_layers + 1):
                if not fixed_preds:
                    self.preds[n] = self.layers[n-1].forward(self.mus[n-1])
                self.errors[n] = self.mus[n] - self.preds[n]


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
