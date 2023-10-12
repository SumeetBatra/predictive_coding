import torch

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import Dataset


def get_mnist_dataset(train=False, download=False):
    dataset = datasets.MNIST(root='./data', train=train, download=download, transform=transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
    ]))

    return dataset


class MNISTDataset(Dataset):
    def __init__(self, train: bool = False, download: bool = False):
        super().__init__()
        self.dataset = get_mnist_dataset(train, download)

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = list(index)
        img, label = self.dataset[index]
        return img, label

    def __len__(self):
        return len(self.dataset)