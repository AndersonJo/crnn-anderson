import os
import re
from argparse import Namespace, ArgumentParser

import torch
from torch import nn

from models.net import AttentionRCNN
from tools.label import LabelConverter
from train import AttentionCRNNModule


def init() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--input', required=True, type=str, help='input image source. (directory or file)')
    parser.add_argument('--output', type=str, help='output path of the inference')
    parser.add_argument('--checkpoint', default='./checkpoints', help='the path of checkpoint directory')
    parser.add_argument('--label', default='./label.txt', help='the path of label txt file')
    parser.add_argument('--backbone', default='resnet50', type=str, help='cnn backbone')
    parser.add_argument('--max_seq', default=8, type=int, help='the maximum sequence length')
    parser.add_argument('--pyramid', dest='pyramid', action='store_true')
    parser.add_argument('--no-pyramid', dest='pyramid', action='store_false')
    parser.add_argument('--cuda', default='cuda')

    # these are required for reasons
    parser.add_argument('--batch', default=16, type=int, help='batch size')
    opt = parser.parse_args()

    parser.set_defaults(pyramid=True, dev=False)

    opt.device = torch.device('cuda' if opt.cuda else 'cpu')
    return opt


def load_model(opt):
    model = None
    if os.path.exists(opt.checkpoint):
        regex = re.compile(r'epoch=(\d+)_val_loss=(\d+\.\d+)\.ckpt')
        checkpoints = [filename for filename in os.listdir(opt.checkpoint) if filename.endswith('.ckpt')]
        assert bool(checkpoints), 'no checkpoint is detected. checkpoint file should ends with .ckpt extension'

        scores = [(ckpt, float(regex.match(ckpt).group(1))) for ckpt in checkpoints]
        scores = sorted(scores, key=lambda x: -x[1])
        checkpoint_path = os.path.join(opt.checkpoint, scores[0][0])

        model = AttentionCRNNModule(opt)

        model.load_state_dict(torch.load(checkpoint_path))
        print('Checkpoint Loaded:', checkpoint_path)

    model.to(opt.device)
    print('is cuda:', next(model.parameters()).is_cuda)
    return model


def main():
    opt = init()
    load_model(opt)


if __name__ == '__main__':
    main()
