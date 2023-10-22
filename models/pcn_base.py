import torch
import torch.nn as nn
import torch.autograd as autograd

from torch.nn.utils.rnn import pad_sequence


class PCNetBase(nn.Module):
    layers: nn.ModuleList
    num_layers: int

    def __init__(self, mu_dt: float, batch_size: int):
        super().__init__()
        self.mu_dt = mu_dt
        self.batch_size = batch_size

    @property
    def params(self):
        return self.layers

    def set_input(self, x: torch.Tensor):
        # clamp the neural activity of 1st layer to the signal
        self.mus[0] = x
        self.mus[0].requires_grad_(True)

    def set_target(self, target: torch.Tensor):
        # clamp the activity of the last layer to be the target
        self.mus[-1] = target
        self.mus[-1].requires_grad_(True)

    def update_grads(self):
        for l in range(self.num_layers):
            mu_pred = self.layers[l].forward(self.layers[l].x)
            delta_w = torch.autograd.grad(mu_pred, inputs=[self.layers[l].weight], grad_outputs=self.errors[l + 1],
                                retain_graph=True)[0]
            delta_b = torch.autograd.grad(mu_pred, inputs=[self.layers[l].bias], grad_outputs=self.errors[l + 1],
                                retain_graph=True)[0]
            self.layers[l].grad['weights'] = delta_w
            self.layers[l].grad['bias'] = delta_b

            # dw, db = torch.autograd.grad(mu_pred, inputs=self.layers[l].parameters(), grad_outputs=self.errors[l + 1],
            #                              retain_graph=True)
            # params = dict(self.layers[l].named_parameters())
            # params['weight'].grad = dw
            # params['bias'].grad = db

    def forward(self, x: torch.Tensor):
        with torch.enable_grad():
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
            self.mus[l].requires_grad_(True)

    def propagate_mu(self):
        for l in range(1, self.num_layers):
            self.mus[l] = self.layers[l-1].forward(self.mus[l-1])

    def inference(self, n_iters: int, fixed_preds: bool = False, predict_label_inputs: bool = False):
        bs = self.mus[0].shape[0]
        for n in range(1, self.num_layers + 1):
            # TODO: I think this is already computed when we call propagate_mu ?
            self.preds[n] = self.layers[n-1].forward(self.mus[n-1])
            self.errors[n] = self.mus[n] - self.preds[n]

        for i in range(n_iters):
            # print(f'Inference iteration {i}')
            if predict_label_inputs:
                # when predict_label_inputs = False, mus[0] is clamped to be the input so there is no delta.
                # however when predict_label_inputs = True, only mus[-1] is clamped to be the target, so we need to
                # compute a delta for mus[0] and update mus[0] as well
                mu_pred = self.layers[0].forward(self.layers[0].x)
                delta = torch.autograd.grad(mu_pred, inputs=[self.layers[0].x], grad_outputs=self.errors[1],
                                            retain_graph=True)[0]
                self.mus[0] = self.mus[0] + self.mu_dt * delta
            # step the activities to minimize free energy
            for l in range(1, self.num_layers):
                # TODO: why is delta computed this way?
                mu_pred = self.layers[l].forward(self.layers[l].x)
                delta = torch.autograd.grad(mu_pred, inputs=[self.layers[l].x], grad_outputs=self.errors[l + 1],
                                    retain_graph=True)[0]
                # TODO: so is mu_d the learning rate for the inference phase?
                self.mus[l] = self.mus[l] + self.mu_dt * delta.view(self.mus[l].shape)

            # recompute error in predictions
            for n in range(1, self.num_layers + 1):
                if not fixed_preds:
                    self.preds[n] = self.layers[n-1].forward(self.mus[n-1])
                self.errors[n] = self.mus[n] - self.preds[n]