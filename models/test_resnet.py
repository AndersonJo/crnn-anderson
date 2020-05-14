import tempfile

import torch
import torch.nn.functional as F

from models.net import load_resnet, PyramidFeatures, AttentionRCNN
from train import AttentionCRNNModule, init


class opt:
    batch: int = 16
    train: str = 'train_data'
    val: str = 'valid_data'
    lr: float = 0.0001
    device = torch.device('cuda')


def test_resnet():
    model = AttentionCRNNModule(opt)
    model.prepare_data()
    val_dl = model.test_dataloader()

    model = AttentionRCNN(device=opt.device)
    model.cuda()
    for i, batch in enumerate(val_dl):
        x, y = batch
        model(x.cuda())

    import ipdb
    ipdb.set_trace()
