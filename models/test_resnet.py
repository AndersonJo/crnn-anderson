import tempfile

import torch
import torch.nn.functional as F

from models.resnet import load_resnet, PyramidFeatures, AttentionRCNN


def test_resnet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attrcnn = AttentionRCNN(device=device)
    attrcnn = attrcnn.cuda()

    x = torch.rand((16, 3, 250, 480)).cuda()
    x2 = attrcnn(x)

    import ipdb
    ipdb.set_trace()
