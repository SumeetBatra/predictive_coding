import torch
import torch.nn as nn


class PCNetBase(nn.Module):
    layers: nn.ModuleList
    num_layers: int

    def __init__(self, mu_dt: float):
        super().__init__()
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

    def propagate_mu(self):
        for l in range(1, self.num_layers):
            self.mus[l] = self.layers[l-1].forward(self.mus[l-1])

    def inference(self, n_iters: int, fixed_preds: bool = False, predict_label_inputs: bool = False):
        bs = self.mus[0].shape[0]
        for n in range(1, self.num_layers + 1):
            # TODO: I think this is already computed when we call propagate_mu ?
            self.preds[n] = self.layers[n-1].forward(self.mus[n-1])
            self.errors[n] = (self.mus[n] - self.preds[n]).view(bs, -1)

        for i in range(n_iters):
            # print(f'Inference iteration {i}')
            if predict_label_inputs:
                # when predict_label_inputs = False, mus[0] is clamped to be the input so there is no delta.
                # however when predict_label_inputs = True, only mus[-1] is clamped to be the target, so we need to
                # compute a delta for mus[0] and update mus[0] as well
                delta = self.layers[0].backward(self.errors[1])
                self.mus[0] = self.mus[0] + self.mu_dt * delta
            # step the activities to minimize free energy
            for l in range(1, self.num_layers):
                # TODO: why is delta computed this way?
                delta = self.layers[l].backward(self.errors[l + 1]).view(bs, -1) - self.errors[l]
                # TODO: so is mu_d the learning rate for the inference phase?
                self.mus[l] = self.mus[l] + self.mu_dt * delta.view(self.mus[l].shape)

            # recompute error in predictions
            for n in range(1, self.num_layers + 1):
                if not fixed_preds:
                    self.preds[n] = self.layers[n-1].forward(self.mus[n-1])
                self.errors[n] = (self.mus[n] - self.preds[n]).view(bs, -1)