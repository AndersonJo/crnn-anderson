from typing import Tuple

import torch


class LabelConverter:

    def __init__(self, label_path: str, max_seq: int, pad='<pad>', delimiter='<delimeter>'):
        self.label_path = label_path
        self.max_seq = max_seq
        self.c2i = dict()
        self.c2i[pad] = 0
        self.c2i[delimiter] = 1

        with open(self.label_path, 'rt') as f:
            for line in f:
                self.c2i[line.strip()] = len(self.c2i)
        self.i2c = {v: k for k, v in self.c2i.items()}

    def to_tensor(self, texts) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = len(texts)

        y_label = torch.zeros(bs, self.max_seq, dtype=torch.int)  # batch, max_seq
        y_seq = torch.zeros(bs, dtype=torch.int)
        for i, text in enumerate(texts):
            for j, c in enumerate(text):
                y_label[i, j] = self.c2i[c]
            y_seq[i] = len(text)

        return y_label, y_seq


    def to_text(self, ):
        pass

    @property
    def n_label(self):
        return len(self.c2i)
