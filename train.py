import os
import re
import shutil
from argparse import ArgumentParser, Namespace
from typing import Union, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim as optim
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import CTCLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import LicensePlateDataset
from models.net import AttentionRCNN
from tools.label import LabelConverter

BASE_DIR = os.path.dirname(__file__)

np.random.seed(128)
torch.manual_seed(128)


def init() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--epoch', default=1000, type=int, help='epoch size')
    parser.add_argument('--batch', default=8, type=int, help='batch size')
    parser.add_argument('--train', default='./train_data', help='the path of train dataset')
    parser.add_argument('--val', default='./valid_data', help='the path of validation dataset')
    parser.add_argument('--label', default='./label.txt', help='the path of label txt file')
    parser.add_argument('--checkpoint', default='./checkpoints', help='the path of checkpoint directory')
    parser.add_argument('--width', default=384, type=int, help='resized image width')
    parser.add_argument('--height', default=256, type=int, help='resized image height')
    parser.add_argument('--backbone', default='resnet101', type=str, help='cnn backbone')
    parser.add_argument('--max_seq', default=8, type=int, help='the maximum sequence length')
    parser.add_argument('--lr', default=0.001, type=int, help='initial learning rate (it uses decay of lr')
    parser.add_argument('--gpu', default=1, type=int, help='the number of gpus')
    parser.add_argument('--cuda', default='cuda')

    opt = parser.parse_args()
    opt.train_path = os.path.join(BASE_DIR, opt.train)
    opt.val_path = os.path.join(BASE_DIR, opt.val)
    opt.label_path = os.path.join(BASE_DIR, opt.label)
    opt.device = torch.device('cuda' if opt.cuda else 'cpu')

    if os.path.exists('lightning_logs'):
        shutil.rmtree('./lightning_logs')
    return opt


class AttentionCRNNModule(pl.LightningModule):

    def __init__(self, opt):
        super(AttentionCRNNModule, self).__init__()

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
        y_text, y_label, y_seq, y_pred, y_pred_seq, loss = self.calculate_loss(batch)
        y_index, pred_text, n_correct, word_acc, char_acc = self.calculate_acc(y_text, y_pred)

        if batch_idx % 20 == 0 and self.logger:
            log_text = '  |  '.join([f'{t2}({t1})' for t1, t2 in zip(y_text[:3], pred_text[:3])])
            self.logger.experiment.add_text('pred_text', log_text)

        tensorboard_logs = {'train_loss': loss,
                            'train_word_acc': word_acc,
                            'train_char_acc': char_acc}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        """
        :param batch: (x, y_test)
            - x     : Tensor(batch, 3, 256, 384)
            - y_text: tuple('Z72모9981', 'Z91오1969', ...)
        """
        y_text, y_label, y_seq, y_pred, y_pred_seq, loss = self.calculate_loss(batch)
        y_index, pred_text, n_correct, word_acc, char_acc = self.calculate_acc(y_text, y_pred)

        tensorboard_logs = {'val_log': loss,
                            'val_word_acc': word_acc,
                            'var_char_acc': char_acc}
        return {'val_loss': loss,
                'val_word_acc': word_acc,
                'val_char_acc': char_acc,
                'log': tensorboard_logs}

    def validation_epoch_end(self, val_outputs):
        avg_loss = torch.stack([x['val_loss'] for x in val_outputs]).mean()
        avg_word_acc = np.mean([x['val_word_acc'] for x in val_outputs])
        avg_char_acc = np.mean([x['val_char_acc'] for x in val_outputs])
        tensorboard_logs = {'val_loss': avg_loss, 'word_acc': avg_word_acc, 'avg_char_acc': avg_char_acc}
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
        return y_text, y_label, y_seq, y_pred, y_pred_seq, loss

    def calculate_acc(self, y_text, y_pred):
        _, y_index = y_pred.max(2)  # y_index: maximum index locations (seq, batch) ex.(192, 8)
        y_index = y_index.transpose(0, 1)
        pred_text = self.label.to_text(y_index)

        n_correct = 0
        c_correct = 0
        n_char = 0
        for t1, t2 in zip(y_text, pred_text):
            if t1 == t2:
                n_correct += 1
            for c1, c2 in zip(t1, t2):
                if c1 == c2:
                    c_correct += 1
                n_char += 1

        word_acc = n_correct / len(y_text)
        char_acc = c_correct / (n_char + 1e-8)
        return y_index, pred_text, n_correct, word_acc, char_acc

    def configure_optimizers(self):
        optimizer = optim.RMSprop(self.parameters(), lr=self.lr)
        lr_scheduler = {'scheduler': ReduceLROnPlateau(optimizer, 'min'),
                        'name': 'ReduceLROnPlateau'}
        return [optimizer], [lr_scheduler]

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


def load_model(opt):
    model = None
    if os.path.exists(opt.checkpoint):
        regex = re.compile(r'epoch=(\d+)_val_loss=(\d+\.\d+)\.ckpt')
        checkpoints = [filename for filename in os.listdir(opt.checkpoint) if filename.endswith('.ckpt')]

        if checkpoints:
            scores = [(ckpt, float(regex.match(ckpt).group(1))) for ckpt in checkpoints]
            scores = sorted(scores, key=lambda x: -x[1])
            checkpoint_path = os.path.join(opt.checkpoint, scores[0][0])
            model = AttentionCRNNModule.load_from_checkpoint(checkpoint_path, opt=opt)
            print('Checkpoint Loaded:', checkpoint_path)

    if model is None:
        model = AttentionCRNNModule(opt)
    model.to(opt.device)
    print('is cuda:', next(model.parameters()).is_cuda)
    return model


def main():
    opt = init()
    model = load_model(opt)

    # Checkpoint
    checkpoint_path = os.path.join(opt.checkpoint, '{epoch:02}_{val_loss:.4f}')
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path,
                                          save_weights_only=True,
                                          save_top_k=1,
                                          verbose=True,
                                          monitor='val_loss',
                                          mode='min')

    # Train
    trainer = Trainer(gpus=opt.gpu,
                      max_epochs=opt.epoch,
                      checkpoint_callback=checkpoint_callback,
                      # track_grad_norm=2,
                      log_gpu_memory=False)
    trainer.fit(model)

    # Development
    # model.prepare_data()
    # test_dl = model.val_dataloader()
    #
    # for i, batch in enumerate(test_dl):
    #     y_pred = model.training_step(batch, i)
    #     break


if __name__ == '__main__':
    main()
