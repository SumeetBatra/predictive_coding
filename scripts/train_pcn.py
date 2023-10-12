import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

from optimizers.optim import Adam
from torch.utils.data import DataLoader
from datasets.mnist_dataset import MNISTDataset
from models.vanilla_pcn import PCNet
from typing import Mapping, List, Any


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num_inference_iters', type=int, default=50)
    parser.add_argument('--test_every', type=int, default=1)
    parser.add_argument('--grad_clip', type=int, default=50)

    args = parser.parse_args()
    return vars(args)


def onehot_encoding(labels: torch.Tensor):
    return F.one_hot(labels)


def accuracy(preds, targets):
    batch_size = preds.shape[0]
    correct = 0
    for b in range(batch_size):
        if torch.argmax(preds) == torch.argmax(targets):
            correct += 1
    return correct / batch_size


def train_supervised(args: Mapping[str, Any]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = args['batch_size']
    t = args['num_inference_iters']

    train_dataset = MNISTDataset(train=True, download=True)
    test_dataset = MNISTDataset(train=False, download=False)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True, drop_last=True)

    model = PCNet(mu_dt=0.01)
    optimizer = Adam(model.params, lr=args['lr'], grad_clip=args['grad_clip'])

    losses = []
    accs = []

    for epoch in range(args['num_epochs']):
        for i, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.view(batch_size, -1).to(device)
            labels = labels.to(device).to(torch.float32)

            optimizer.zero_grad()
            # reset activity predictions, prediction errors, and true activities
            model.reset()
            # clamp first activity to be the data
            model.set_input(labels)
            # forward propagate the activity due to the input signal
            model.propogate_mu()
            # clamp the output activity to be the labels for the input data
            model.set_target(labels)
            # perform inference to minimize the free energy
            model.inference(n_iters=t)
            # update the weights
            optimizer.step(
                curr_epoch=epoch,
                curr_batch=i,
                n_batches=len(train_loader),
                batch_size=imgs.shape[0]
            )

            if epoch % args['test_every'] == 0:
                acc = 0
                for _, (imgs, labels) in enumerate(test_loader):
                    label_preds = model(imgs)
                    acc += accuracy(label_preds, labels)
                print(f'Test @ Epoch {epoch} / Accuracy: {(acc / len(test_loader)):.4f}')


if __name__ == '__main__':
    args = parse_args()
    train_supervised(args)
