from typing import Tuple

import torch


class LabelConverter:

    def __init__(self, label_path: str, max_seq: int, pad=' ', delimiter='<delimeter>'):
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

    def to_text(self, y_index: torch.Tensor, y_seqs: torch.Tensor = None):
        """
        :param y_index: _, y_index = y_pred.max(2)
        :param y_seqs: a list of sequence length (batch_size,) ex.(192, 192, 192, ... 192)
        :return:
        """
        seq_size, batch_size = y_index.shape

        if y_seqs is None:
            y_seqs = torch.LongTensor([seq_size] * batch_size)

        assert y_index.numel() == y_seqs.sum()
        texts = []

        for i in range(batch_size):
            n_seq = y_seqs[i]  # 192
            sequence = y_index[:, i]
            text = []
            for j in range(n_seq):
                if sequence[j] != 0 and (not (j > 0 and sequence[j - 1] == sequence[j])):
                    text.append(self.i2c[sequence[j].item()])
            texts.append(text)

        return texts

    @property
    def n_label(self):
        return len(self.c2i)
