import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import wandb
import matplotlib.pyplot as plt

from pathlib import Path
from optimizers.optim import Adam
from torch.utils.data import DataLoader
from datasets.mnist_dataset import MNISTDataset
from models.vanilla_pcn import PCNet, GenerativePCNet
from typing import Mapping, List, Any
from common.utils import config_wandb
from distutils.util import strtobool


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num_inference_iters', type=int, default=50)
    parser.add_argument('--test_every', type=int, default=1)
    parser.add_argument('--grad_clip', type=int, default=50)
    parser.add_argument('--generative', type=lambda x: bool(strtobool(x)), default=False,
                        help='Use PCN as a generative model (or associative memory model)')
    # wandb settings
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--wandb_project', type=str, default='pcn')
    parser.add_argument('--wandb_run_name', type=str, default='pcn_mnist_test')
    parser.add_argument('--wandb_group', type=str, default='mnist')
    parser.add_argument('--wandb_entity', type=str, default='qdrl')
    parser.add_argument('--wandb_tag', type=str, default='supervised')
    # generative training settings
    parser.add_argument('--exp_dir', type=str, default='./results')

    args = parser.parse_args()
    return vars(args)


def accuracy(preds, targets):
    batch_size = preds.shape[0]
    correct = (torch.argmax(preds, dim=1) == torch.argmax(targets, dim=1)).sum()
    return correct / batch_size


def train(args: Mapping[str, Any]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = args['batch_size']
    test_batch_size = args['test_batch_size']
    t = args['num_inference_iters']
    generative = args['generative']

    # setup dirs to save generated images if training generative model
    exp_dir = Path(args['exp_dir'])
    exp_dir.mkdir(exist_ok=True)
    img_dir = exp_dir.joinpath('images')
    img_dir.mkdir(exist_ok=True)

    train_dataset = MNISTDataset(train=True, download=True)
    test_dataset = MNISTDataset(train=False, download=False)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True, drop_last=True)

    model = GenerativePCNet(mu_dt=0.01) if args['generative'] else PCNet(mu_dt=0.01)
    model.to(device)
    optimizer = Adam(model.params, lr=args['lr'], grad_clip=args['grad_clip'], batch_scale=False)

    losses = []
    accs = []

    for epoch in range(args['num_epochs']):
        for i, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.view(batch_size, -1).to(device)
            labels = F.one_hot(labels, num_classes=10).to(device).to(torch.float32)

            optimizer.zero_grad()
            # reset activity predictions, prediction errors, and true activities
            model.reset()
            # clamp first activity to be the data
            input = labels if generative else imgs
            model.set_input(input)
            # forward propagate the activity due to the input signal
            model.propagate_mu()
            # clamp the output activity
            target = imgs if generative else labels
            model.set_target(target)
            # perform inference to minimize the free energy
            fixed_preds = False if generative else True
            model.inference(n_iters=t, fixed_preds=fixed_preds)
            # manually set the gradients
            model.update_grads()
            # update the weights
            optimizer.step(
                curr_epoch=epoch,
                curr_batch=i,
                n_batches=len(train_loader),
                batch_size=imgs.shape[0]
            )

        if epoch % args['test_every'] == 0:
            if not args['generative']:
                acc = 0
                for _, (imgs, labels) in enumerate(test_loader):
                    imgs = imgs.view(batch_size, -1).to(device)
                    labels = F.one_hot(labels, num_classes=10).to(device).to(torch.float32)
                    label_preds = model(imgs)
                    acc += accuracy(label_preds, labels)
                epoch_acc = 100. * acc / len(test_loader)
                print(f'Test @ Epoch {epoch} / Accuracy: {epoch_acc:.4f}')
                if args['use_wandb']:
                    wandb.log({
                        'epoch': epoch,
                        'test/accuracy': epoch_acc,
                    })
            else:
                acc = 0
                for _, (imgs, labels) in enumerate(test_loader):
                    imgs = imgs.view(test_batch_size, -1).to(device)
                    labels = F.one_hot(labels, num_classes=10).to(device).to(torch.float32)
                    ### recover labels given the image data
                    # reset the predictions, mus, and errors
                    model.reset()
                    # reset the layer outputs randomly
                    model.reset_mus(batch_size=test_batch_size, init_std=0.01)
                    # clamp last layer output to be the images
                    model.set_target(imgs)
                    # perform inference
                    model.inference(n_iters=200, predict_label_inputs=True)
                    label_preds = model.mus[0]
                    acc += accuracy(label_preds, labels)
                epoch_acc = 100. * acc / len(test_loader)
                print(f'Test @ Epoch {epoch} / Accuracy: {epoch_acc:.4f}')
                if args['use_wandb']:
                    wandb.log({
                        'epoch': epoch,
                        'test/accuracy': epoch_acc,
                    })

                ### generate the images given the labels
                model.reset()
                _, labels = next(iter(test_loader))
                labels = labels[:8]
                labels = F.one_hot(labels, num_classes=10).to(device).to(torch.float32)
                img_preds = model.forward(labels).cpu().detach().reshape(-1, 28, 28)
                _, axes = plt.subplots(2, 4)
                axes = axes.flatten()
                for i, img in enumerate(img_preds):
                    axes[i].imshow(img, cmap='gray')
                save_path = img_dir.joinpath(f'generated_images_epoch_{epoch}.png')
                plt.savefig(save_path)
                plt.close('all')


if __name__ == '__main__':
    args = parse_args()

    if args['use_wandb']:
        # setup wandb
        config_wandb(wandb_project=args['wandb_project'],
                     entity=args['wandb_entity'],
                     wandb_group=args['wandb_group'],
                     run_name=args['wandb_run_name'],
                     tags=args['wandb_tag'],
                     cfg=args)

    train(args)
