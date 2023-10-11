import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

from datasets.mnist_dataset import MNISTDataset
from models.vanilla_pcn import PCNet
from typing import Mapping, List, Any


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.003)

    args = parser.parse_args()
    return vars(args)


def onehot_encoding(labels: torch.Tensor):
    return F.one_hot(labels)


def train(args: Mapping[str, Any]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = MNISTDataset(train=True, download=True)

    model = PCNet

    loss_fn = nn.MSELoss

    batch_size = args['batch_size']

    losses = []
    accs = []

    for n in range(args['num_epochs']):
        for i, (img, labels) in enumerate(dataset):
            img = img.view(batch_size, -1).to(device)
            onehot_labels = onehot_encoding(labels).to(device)

            # perform inference using backprop



if __name__ == '__main__':
    args = parse_args()
    train(args)
