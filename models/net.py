from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from torchvision.models import ResNet as TorchVisionResnet
from torchvision.models.resnet import model_urls, Bottleneck


class AttentionRCNN(nn.Module):
    def __init__(self, device: torch.device, backbone: str = 'resnet101', hidden_size=64, n_class=10):
        super(AttentionRCNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.resnet = load_resnet(backbone)
        self.pyramid = PyramidFeatures(*self.resnet.fpn_sizes)

        self.encoder = Encoder(128, self.hidden_size, device=self.device)
        self.decoder = Decoder(self.hidden_size * 2, self.hidden_size, n_class, device=self.device)

    def forward(self, x):
        batch_size = x.size(0)

        # Resnet & PyramidNet
        h = self.resnet(x)
        features = self.pyramid(h)
        f_dim = features[0].size(2) * features[0].size(3)
        cnn_feature = torch.cat([feature.view(batch_size, -1, f_dim) for feature in features], dim=1)

        encoder_output, hiddens = self.encoder(cnn_feature)
        h_0, c_0 = hiddens[-1]
        h_s = torch.cat([h[0] for h in hiddens])
        c_s = torch.cat([h[1] for h in hiddens])

        decoder_output = self.decoder(cnn_feature, encoder_output, (h_0, c_0), h_s, c_s)

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
        self.P5_avg_pool = nn.AdaptiveAvgPool2d((8, 16))

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P4_avg_pool = nn.AdaptiveAvgPool2d((8, 16))

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P3_avg_pool = nn.AdaptiveAvgPool2d((8, 16))

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)
        self.P6_avg_pool = nn.AdaptiveAvgPool2d((8, 16))

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)
        self.P7_avg_pool = nn.AdaptiveAvgPool2d((8, 16))

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

        return [P3_y, P5_y, P7_y]


class Encoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, device):
        super(Encoder, self).__init__()

        self.batch_size = 1
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device

        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, dropout=0.1)

    def forward(self, cnn_feature):
        """
        :param x: (batch_size, seq_size, input_size)
        :return:
        """
        bs, ss, hs = cnn_feature.shape
        h_0, c_0 = self.init_hidden(batch_size=bs)

        cnn_feature = cnn_feature.permute(1, 0, 2)
        encoder_outputs = []
        hiddens = []
        for encoder_input in cnn_feature:
            encoder_output, (h_0, c_0) = self.lstm(encoder_input.unsqueeze(0), (h_0, c_0))
            encoder_outputs.append(encoder_output)
            hiddens.append((h_0, c_0))
        return torch.cat(encoder_outputs), hiddens

    def init_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # h_0: (n_layers * n_directions, batch, hidden_size)
        # c_0: (n_layers * n_directions, batch, hidden_size)
        h_0 = torch.zeros((2, batch_size, self.hidden_size), device=self.device)
        c_0 = torch.zeros((2, batch_size, self.hidden_size), device=self.device)
        return h_0, c_0


class Decoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_class, device: torch.device):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_class = n_class
        self.device = device

        self.dropout = nn.Dropout(0.3)

        self.linear_h = nn.Linear(self.hidden_size, 32)
        self.linear_c = nn.Linear(self.hidden_size, 32)
        self.lstm1 = BidirectionalLSTM(128, 32, 32)
        self.lstm2 = BidirectionalLSTM(32, 32, n_class)

    def forward(self, cnn_feature: torch.Tensor, encoder_output: torch.Tensor,
                hidden: Tuple[torch.Tensor, torch.Tensor], h_s: torch.Tensor, c_s: torch.Tensor):
        cnn_feature = cnn_feature.permute(1, 0, 2)
        cnn_ss, cnn_bs, cnn_ds = cnn_feature.shape

        h_0, c_0 = hidden
        h_0 = self.linear_h(h_0)
        c_0 = self.linear_c(c_0)

        cnn_feature = self.dropout(cnn_feature)
        out1, (h_1, c_1) = self.lstm1(cnn_feature, h_0, c_0)
        out2, (h_1, c_1) = self.lstm2(out1, h_1, c_1)
        output = F.log_softmax(out2, dim=2)
        return output


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
