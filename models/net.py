from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.hub import load_state_dict_from_url
from torchvision.models import ResNet as TorchVisionResnet, vgg19_bn
from torchvision.models.resnet import model_urls, Bottleneck


class AttentionRCNN(nn.Module):
    def __init__(self, backbone: str, n_class: int, use_pyramid: bool, device: torch.device, hidden_size=256):
        super(AttentionRCNN, self).__init__()
        self.use_pyramid = use_pyramid
        self.device = device
        self.hidden_size = hidden_size
        self.resnet = load_resnet(backbone)
        self.pyramid = PyramidFeatures(*self.resnet.fpn_sizes)

        if self.use_pyramid:
            self.decoder = Decoder(768, 384, n_class, device=self.device)
        else:
            self.decoder = Decoder(1024, 384, n_class, device=self.device)

    def forward(self, x):
        batch_size = x.size(0)

        # Resnet & PyramidNet
        h = self.resnet(x)

        if self.use_pyramid:
            # Pyramid Net (Deconvolution Network)
            cnn_feature = self.pyramid(h)
            cnn_feature = cnn_feature.permute(2, 0, 1)  # (107, 16, 512)

        else:
            # No Pyramid
            conv = h[2]  # (batch, 2048, 8, 12)
            bs, fs, hs, ws = conv.shape
            cnn_feature = conv.view(bs, fs, hs * ws)  # (batch, cnn_feature, w*h) ex.(batch, 2048, 96)
            cnn_feature = cnn_feature.permute(2, 0, 1)  # (width* height, batch, cnn_feature) ex.(96, 32, 2048)

        decoder_output = self.decoder(cnn_feature)
        return decoder_output


class ResNet(TorchVisionResnet):

    def __init__(self, block, layers, *args, **kwargs):
        super(ResNet, self).__init__(block, layers, *args, **kwargs)

        self.fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels,
                          self.layer3[layers[2] - 1].conv3.out_channels,
                          self.layer4[layers[3] - 1].conv3.out_channels]

    def _forward_impl(self, x):
        h1 = self.conv1(x)
        h2 = self.bn1(h1)
        h3 = self.relu(h2)
        h4 = self.maxpool(h3)

        h5 = self.layer1(h4)
        h6 = self.layer2(h5)
        h7 = self.layer3(h6)
        h8 = self.layer4(h7)

        return h5, h6, h7, h8


class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=64):
        super(PyramidFeatures, self).__init__()

        self.c5_deconv = nn.Conv2d(256, 768, 1)
        self.c6_deconv = nn.Conv2d(512, 768, 1)
        self.c7_deconv = nn.Conv2d(1024, 768, 1)
        self.c8_deconv = nn.Conv2d(2048, 768, 1)

    def forward(self, inputs):
        """
        CNN features
         - c5: (batch, 256, 24, 40)
         - c6: (batch, 512, 12, 20)
         - c7: (batch, 1024, 6, 10)
         - c8: (batch, 2048, 3, 5)
        """
        c5, c6, c7, c8 = inputs
        batch, ss, hw, ws = c6.shape

        c5_h = self.c5_deconv(c5).view(batch, 768, -1)
        c6_h = self.c6_deconv(c6).view(batch, 768, -1)
        c7_h = self.c7_deconv(c7).view(batch, 768, -1)
        c8_h = self.c8_deconv(c8).view(batch, 768, -1)
        output = torch.cat([c5_h, c6_h, c7_h, c8_h], dim=2)  # (batch, 512, 315)
        return output


class Decoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_class, device: torch.device):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_class = n_class
        self.device = device

        self.dropout = nn.Dropout(0.1)

        self.lstm1 = BidirectionalLSTM(self.input_size, self.hidden_size, self.hidden_size)
        self.lstm2 = BidirectionalLSTM(self.hidden_size, self.hidden_size, n_class)

    def forward(self, cnn_feature: torch.Tensor):
        ss, bs, hs = cnn_feature.shape

        h_0, c_0 = self.init_hidden(batch_size=bs)
        out1, (h_1, c_1) = self.lstm1(cnn_feature, (h_0, c_0))
        out2, (h_1, c_1) = self.lstm2(out1, (h_1, c_1))
        output = F.log_softmax(out2, dim=2)
        return output

    def init_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # h_0: (n_layers * n_directions, batch, hidden_size)
        # c_0: (n_layers * n_directions, batch, hidden_size)
        h_0 = torch.zeros((2, batch_size, self.hidden_size), device=self.device)
        c_0 = torch.zeros((2, batch_size, self.hidden_size), device=self.device)
        return h_0, c_0


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input, hidden=None):
        recurrent, (h_1, c_1) = self.rnn(input, hidden)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output, (h_1, c_1)


def load_resnet(backbone: str = 'resnet50', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    state_dict = load_state_dict_from_url(model_urls[backbone], progress=True)
    model.load_state_dict(state_dict)
    return model
