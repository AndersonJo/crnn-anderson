import os
from argparse import ArgumentParser, Namespace
from typing import Union, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch.nn import CTCLoss
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import LicensePlateDataset
from models.net import AttentionRCNN
from tools.label import LabelConverter
from pytorch_lightning import Trainer

BASE_DIR = os.path.dirname(__file__)

np.random.seed(128)
torch.manual_seed(128)


def init() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--batch', default=8, type=int, help='batch size')
    parser.add_argument('--train', default='./train_data', help='the path of train dataset')
    parser.add_argument('--val', default='./valid_data', help='the path of validation dataset')
    parser.add_argument('--label', default='./label.txt', help='the path of label txt file')
    parser.add_argument('--width', default=384, type=int, help='resized image width')
    parser.add_argument('--height', default=256, type=int, help='resized image height')
    parser.add_argument('--backbone', default='resnet101', type=str, help='cnn backbone')
    parser.add_argument('--max_seq', default=8, type=int, help='the maximum sequence length')
    parser.add_argument('--lr', default=0.0001, type=int, help='learning rate')
    parser.add_argument('--cuda', default='cuda')

    opt = parser.parse_args()
    opt.train_path = os.path.join(BASE_DIR, opt.train)
    opt.val_path = os.path.join(BASE_DIR, opt.val)
    opt.label_path = os.path.join(BASE_DIR, opt.label)
    opt.device = torch.device('cuda' if opt.cuda else 'cpu')
    return opt


class AttentionCRNNModel(pl.LightningModule):

    def __init__(self, opt):
        super(AttentionCRNNModel, self).__init__()

        self.batch: int = opt.batch
        self.img_height = opt.height
        self.img_width = opt.width
        self.train_path: str = opt.train
        self.valid_path: str = opt.val
        self.lr: float = opt.lr
        self.device = opt.device

        self.label = LabelConverter(opt.label_path, opt.max_seq)
        self.n_label: int = self.label.n_label

        self.train_dataset: Union[LicensePlateDataset, None] = None
        self.test_dataset: Union[LicensePlateDataset, None] = None
        self._transform = None

        self.model = AttentionRCNN(backbone=opt.backbone,
                                   n_class=self.n_label,
                                   device=opt.device)
        self.criterion = CTCLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        :param batch: (x, y_test)
            - x     : Tensor(batch, 3, 256, 384)
            - y_text: tuple('Z72모9981', 'Z91오1969', ...)
        """
        y_label, y_seq, y_pred, y_pred_seq, loss = self.calculate_loss(batch)

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        """
        :param batch: (x, y_test)
            - x     : Tensor(batch, 3, 256, 384)
            - y_text: tuple('Z72모9981', 'Z91오1969', ...)
        """
        y_label, y_seq, y_pred, y_pred_seq, loss = self.calculate_loss(batch)
        tensorboard_logs = {'val_log': loss}
        return {'val_loss': loss, 'log': tensorboard_logs}

    def validation_epoch_end(self, val_outputs):
        avg_loss = torch.stack([x['val_loss'] for x in val_outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def calculate_loss(self, batch: Tuple[torch.Tensor, tuple]):
        x, y_text = batch
        batch_size = x.size(0)
        # Preprocess y_label
        y_label, y_seq = self.label.to_tensor(y_text)

        # Inference
        y_pred = self(x.to(self.device))  # (192, 16, 66)

        # Calculate loss
        y_pred_seq = torch.LongTensor([y_pred.size(0)] * batch_size)
        loss = self.criterion(y_pred, y_label, y_pred_seq, y_seq) / batch_size
        return y_label, y_seq, y_pred, y_pred_seq, loss

    def configure_optimizers(self):
        return optim.RMSprop(self.parameters(), lr=self.lr)

    def prepare_data(self):
        self.train_dataset = LicensePlateDataset(self.train_path, transform=self.transform_compose)
        self.test_dataset = LicensePlateDataset(self.valid_path, transform=self._transform)

    def train_dataloader(self) -> DataLoader:
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch, shuffle=True, num_workers=4)
        return train_loader

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        val_loader = DataLoader(self.test_dataset, batch_size=self.batch, shuffle=False, num_workers=4)
        return val_loader

    # def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
    #     test_loader = DataLoader(self.test_dataset, batch_size=self.batch, shuffle=True, num_workers=4)
    #     return test_loader

    @property
    def transform_compose(self) -> transforms.Compose:
        if self._transform is not None:
            return self._transform
        _mean = (114.35613348, 135.84955821, 105.15833282)
        _std = (62.10358157, 51.18792043, 56.16809703)
        transform = transforms.Compose([
            transforms.Resize((self.img_height, self.img_width)),
            transforms.RandomVerticalFlip(0.3),
            transforms.RandomGrayscale(0.1),
            transforms.ColorJitter(brightness=(0.2, 2), contrast=(0.3, 2), saturation=(0.2, 2), hue=(-0.3, 0.3)),
            transforms.ToTensor(),
            transforms.Normalize(_mean, _std)
        ])
        self._transform = transform
        return self._transform


def main():
    opt = init()

    model = AttentionCRNNModel(opt)
    model.to(opt.device)
    print('is cuda:', next(model.parameters()).is_cuda)

    trainer = Trainer(gpus=1, max_epochs=10)
    trainer.fit(model)
    # model.prepare_data()
    # test_dl = model.test_dataloader()
    # for i, batch in enumerate(test_dl):
    #     y_pred = model.training_step(batch, i)


if __name__ == '__main__':
    main()
