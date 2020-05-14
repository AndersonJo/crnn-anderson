from typing import Tuple

import numpy as np
from argparse import ArgumentParser, Namespace

import torch

from dataset import LicensePlateDataset
from torchvision import transforms
from torch.utils.data import DataLoader

np.random.seed(128)
torch.manual_seed(128)


def init() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--train', default='./train_data', help='the path of train dataset')
    parser.add_argument('--val', default='./valid_data', help='the path of validation dataset')
    parser.add_argument('--cuda', default='cuda')

    opt = parser.parse_args()
    opt.device = torch.device('cuda' if opt.cuda else 'cpu')
    return opt


def data_loader(train_path: str, val_path: str) -> Tuple[DataLoader, DataLoader]:
    _mean = (114.35613348, 135.84955821, 105.15833282)
    _std = (62.10358157, 51.18792043, 56.16809703)
    transform = transforms.Compose([
        transforms.Resize((250, 400)),
        transforms.RandomVerticalFlip(0.3),
        transforms.RandomGrayscale(0.1),
        transforms.ColorJitter(brightness=(0.2, 2), contrast=(0.3, 2), saturation=(0.2, 2), hue=(-0.3, 0.3)),
        transforms.ToTensor(),
        transforms.Normalize(_mean, _std)
    ])
    train = LicensePlateDataset(train_path, transform=transform)
    val = LicensePlateDataset(val_path, transform=transform)
    train_loader = DataLoader(train, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val, batch_size=32, shuffle=True, num_workers=4)

    return train_loader, val_loader


def main():
    opt = init()
    train_loader = data_loader(opt.train, opt.val)


if __name__ == '__main__':
    main()
