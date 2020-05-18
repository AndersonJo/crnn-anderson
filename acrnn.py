import os
import re
import shutil
from argparse import ArgumentParser, Namespace
from datetime import datetime
from typing import Union, List, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.optim as optim
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from skimage import io
from torch.nn import CTCLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import LicensePlateDataset
from models.net import AttentionRCNN
from tools.label import LabelConverter

BASE_DIR = os.path.dirname(__file__)

np.random.seed(128)
torch.manual_seed(128)


def init() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('mode', default='inference', help='inference or train')
    # Inference
    parser.add_argument('--input', default='./valid_data', help='inference input (directory)')
    parser.add_argument('--output', default='./output', help='inference output (directory)')

    # Train and Inference
    parser.add_argument('--epoch', default=1000, type=int, help='epoch size')
    parser.add_argument('--batch', default=16, type=int, help='batch size')
    parser.add_argument('--train', default='./train_data', help='the path of train dataset')
    parser.add_argument('--val', default='./valid_data', help='the path of validation dataset')
    parser.add_argument('--label', default='./label.txt', help='the path of label txt file')
    parser.add_argument('--checkpoint', default='./checkpoints', help='the path of checkpoint directory')
    parser.add_argument('--width', default=120, type=int, help='resized image width')
    parser.add_argument('--height', default=40, type=int, help='resized image height')
    parser.add_argument('--backbone', default='resnet50', type=str, help='cnn backbone')
    parser.add_argument('--max_seq', default=8, type=int, help='the maximum sequence length')
    parser.add_argument('--lr', default=0.0001, type=int, help='initial learning rate (it uses decay of lr')
    parser.add_argument('--gpu', default=1, type=int, help='the number of gpus')
    parser.add_argument('--pyramid', dest='pyramid', action='store_true')
    parser.add_argument('--no-pyramid', dest='pyramid', action='store_false')
    parser.add_argument('--dev', dest='dev', action='store_true')
    parser.add_argument('--cuda', default='cuda')

    parser.set_defaults(pyramid=True, dev=False)

    opt = parser.parse_args()
    opt.train_path = os.path.join(BASE_DIR, opt.train)
    opt.val_path = os.path.join(BASE_DIR, opt.val)
    opt.label_path = os.path.join(BASE_DIR, opt.label)
    opt.device = torch.device('cuda' if opt.cuda else 'cpu')

    if os.path.exists('lightning_logs'):
        shutil.rmtree('./lightning_logs')

    print('Backbone:', opt.backbone)
    print('Pyramid Net:', opt.pyramid)
    print('Development:', opt.dev)
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
        self.dev = opt.dev

        self.label = LabelConverter(opt.label_path, opt.max_seq)
        self.n_label: int = self.label.n_label

        self.train_dataset: Union[LicensePlateDataset, None] = None
        self.test_dataset: Union[LicensePlateDataset, None] = None
        self._transform = None

        self.model = AttentionRCNN(backbone=opt.backbone,
                                   n_class=self.n_label,
                                   use_pyramid=opt.pyramid,
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
            log_text = '  |  '.join([f'{t2}({t1})' for t1, t2 in zip(y_text, pred_text)])
            self.logger.experiment.add_text('train_pred_text', log_text)

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

        if batch_idx % 10 == 0 and self.logger:
            log_text = '  |  '.join([f'{t2}({t1})' for t1, t2 in zip(y_text, pred_text)])
            self.logger.experiment.add_text('val_pred_text', log_text)

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
        tensorboard_logs = {'val_loss': avg_loss, 'val_word_acc': avg_word_acc, 'avg_char_acc': avg_char_acc}
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
                if c1 == c2 and not (c1 == '-' or c2 == '-'):
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
        self.train_dataset = LicensePlateDataset(self.train_path, transform=self.get_transform(), dev=self.dev)
        self.test_dataset = LicensePlateDataset(self.valid_path, transform=self.get_transform(), dev=self.dev)

    def train_dataloader(self) -> DataLoader:
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch, shuffle=True, num_workers=4)
        return train_loader

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        val_loader = DataLoader(self.test_dataset, batch_size=self.batch, shuffle=False, num_workers=4)
        return val_loader

    # def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
    #     test_loader = DataLoader(self.test_dataset, batch_size=self.batch, shuffle=True, num_workers=4)
    #     return test_loader

    def get_transform(self) -> transforms.Compose:
        if self._transform is not None:
            return self._transform
        # _mean = (114.35613348, 135.84955821, 105.15833282)
        # _std = (62.10358157, 51.18792043, 56.16809703)
        transform = transforms.Compose([
            transforms.Resize((self.img_height, self.img_width)),
            transforms.RandomGrayscale(0.1),
            transforms.ColorJitter(brightness=(0.2, 2), contrast=(0.3, 2), saturation=(0.2, 2), hue=(-0.3, 0.3)),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])
        if self.dev:
            transform = transforms.Compose([
                transforms.Resize((self.img_height, self.img_width)),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5)
            ])
        self._transform = transform
        return self._transform


def load_model(opt) -> AttentionCRNNModule:
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


def train(opt, model):
    # Checkpoint
    checkpoint_path = os.path.join(opt.checkpoint, '{epoch:02}_{val_loss:.4f}')
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path,
                                          save_weights_only=True,
                                          save_top_k=1,
                                          verbose=True,
                                          monitor='val_loss',
                                          mode='min')

    # Train
    val_percent_check = 0.1 if opt.dev else 1.
    trainer = Trainer(gpus=opt.gpu,
                      max_epochs=opt.epoch,
                      checkpoint_callback=checkpoint_callback,
                      val_percent_check=val_percent_check,
                      # track_grad_norm=2,
                      log_gpu_memory=False)
    trainer.fit(model)


def inference(opt, model):
    shutil.rmtree(opt.output)

    if not os.path.exists(opt.output):
        os.mkdir(opt.output)

    # DataLoader
    device = opt.device
    transform = transforms.Compose([
        transforms.Resize((opt.height, opt.width)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    test_dataset = LicensePlateDataset(opt.input, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch, shuffle=True, num_workers=4)

    start_dt = datetime.now()
    results = []
    for batch in tqdm(test_dataloader, desc='Batch'):
        images, y_text = batch

        y_pred = model(images.to(device))
        _, y_index = y_pred.max(2)  # y_index: maximum index locations (seq, batch) ex.(192, 8)
        y_index = y_index.transpose(0, 1)
        pred_text = model.label.to_text(y_index)

        # Save results to outputs
        for t1, t2 in zip(y_text, pred_text):
            row = {}
            row['y_true'] = t1
            row['y_pred'] = t2
            row['is_correct'] = t1 == t2
            row['n_correct_char'] = sum([a == b for a, b in zip(t1, t2)])
            results.append(row)

            # Save image
            if np.random.rand() > 0.95 or not row['is_correct']:
                from_filename = os.path.join(opt.input, f'{t1}.jpg')
                original_image = Image.fromarray(io.imread(from_filename))
                to_filename = os.path.join(opt.output, f'{"o" if row["is_correct"] else "x"}_{t2}_({t1}).jpg')
                original_image.save(to_filename)

    results = pd.DataFrame(results)
    test_size = results.shape[0]
    acc_word = results['is_correct'].mean()
    acc_char = results['n_correct_char'].mean()
    taken_dt = datetime.now() - start_dt
    print(f'test_size        : {test_size}')
    print(f'total taken time :', taken_dt)
    print(f'time per image   :', taken_dt / test_size)
    print(f'n correct word   :', results['is_correct'].sum())
    print(f'n incorrect word :', (~results['is_correct']).sum())
    print(f'word accuracy    : {acc_word:.8f}')
    print(f'char accuracy    : {acc_char:.8f}')
    print(results.describe())


def main():
    opt = init()
    model = load_model(opt)

    # Train
    if opt.mode == 'train':
        train(opt, model)

    else:
        inference(opt, model)

    # Development
    # model.prepare_data()
    # test_dl = model.val_dataloader()
    #
    # for i, batch in enumerate(test_dl):
    #     y_pred = model.training_step(batch, i)
    #     break


if __name__ == '__main__':
    main()
