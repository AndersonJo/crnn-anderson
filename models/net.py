from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from torchvision.models import ResNet as TorchVisionResnet
from torchvision.models.resnet import model_urls, Bottleneck


class AttentionRCNN(nn.Module):
    def __init__(self, backbone: str, n_class: int, device: torch.device, hidden_size=64):
        super(AttentionRCNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.resnet = load_resnet(backbone)
        self.pyramid = PyramidFeatures(*self.resnet.fpn_sizes)

        # self.encoder = Encoder(128, self.hidden_size, device=self.device)
        self.decoder = Decoder(2048, 256, n_class, device=self.device)

    def forward(self, x):
        batch_size = x.size(0)

        # Resnet & PyramidNet
        h = self.resnet(x)
        conv = h[-1]  #  (batch, 2048, 8, 12)
        bs, fs, hs, ws = conv.shape
        conv = conv.view(bs, fs, hs*ws)  # (batch, 2048, 96)

        # features = self.pyramid(h)
        # cnn_feature = torch.cat([f.squeeze(2) for f in features], dim=2)  # (batch, 64, 768)

        decoder_output = self.decoder(conv)
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

        return h6, h7, h8


class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=64):
        super(PyramidFeatures, self).__init__()

        self.C3_size = C3_size
        self.C4_size = C4_size
        self.C5_size = C5_size

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P5_avg_pool = nn.AdaptiveAvgPool2d((1, 256))

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P4_avg_pool = nn.AdaptiveAvgPool2d((1, 128))

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P3_avg_pool = nn.AdaptiveAvgPool2d((1, 128))

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)
        self.P6_avg_pool = nn.AdaptiveAvgPool2d((1, 128))

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)
        self.P7_avg_pool = nn.AdaptiveAvgPool2d((1, 128))

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_h = self.P5_1(C5)
        P5_h = self.P5_upsampled(P5_h)
        P5_h = self.P5_2(P5_h)
        P5_y = self.P5_avg_pool(P5_h)

        P4_h = self.P4_1(C4)
        P4_h = P5_h + P4_h
        P4_h = self.P4_upsampled(P4_h)
        P4_x = self.P4_2(P4_h)
        P4_y = self.P4_avg_pool(P4_x)

        P3_h = self.P3_1(C3)
        P3_h = P3_h + P4_h
        P3_x = self.P3_2(P3_h)
        P3_y = self.P3_avg_pool(P3_x)

        P6_x = self.P6(C5)
        P6_y = self.P6_avg_pool(P6_x)

        P7_x = self.P7_2(self.P7_1(P6_x))
        P7_y = self.P7_avg_pool(P7_x)

        return [P3_y, P4_y, P5_y, P6_y, P7_y]


class Decoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_class, device: torch.device):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_class = n_class
        self.device = device

        self.dropout = nn.Dropout(0.1)

        self.lstm1 = BidirectionalLSTM(self.input_size, 256, 256)
        self.lstm2 = BidirectionalLSTM(256, 256, n_class)

    def forward(self, cnn_feature: torch.Tensor):
        cnn_feature = cnn_feature.permute(2, 0, 1)
        cnn_feature = self.dropout(cnn_feature)
        ss, bs, hs = cnn_feature.shape

        h_0, c_0 = self.init_hidden(batch_size=bs)
        out1, (h_1, c_1) = self.lstm1(cnn_feature, h_0, c_0)
        out2, (h_1, c_1) = self.lstm2(out1, h_1, c_1)
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

    def forward(self, input, h_0, c_0):
        recurrent, (h_1, c_1) = self.rnn(input, (h_0, c_0))
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output, (h_1, c_1)


def load_resnet(backbone: str = 'resnet101', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    state_dict = load_state_dict_from_url(model_urls[backbone], progress=True)
    model.load_state_dict(state_dict)
    return model
