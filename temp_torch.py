import argparse
import os
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets
import torchvision.transforms as transforms
import torchvision.utils as utils
import numpy as np
import matplotlib.pyplot as plt

def trainset(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
    dataset = torchvision.datasets.MNIST(
        root=args.root,train=True,download=True,transform=transform
        )
    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,shuffle=True, 
        num_workers=args.workers
        )
    device = torch.device('cuda' if args.cuda else 'cpu')

    return trainloader, device

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(args.noise, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh())

    def forward(self, x):
        return self.gen(x)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_cuda', action='store_true', default=True)
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--seed', type=int, default=999)
    parser.add_argument('--workers', type=int, defualt=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--noise_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--num_visualize', type=int, default=16)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    trainloader, device = trainset(args)
