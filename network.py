import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTM(nn.Module):
    def __init__(self, num_words, input_size=1024, hidden_size=2048, num_layer=3):
        super(LSTM, self).__init__()
        self.emd = nn.Embedding(num_words, input_size)
        self.net = nn.LSTM(input_size, hidden_size, num_layer, batch_first=True, bidirectional=False)
        self.classification = nn.Sequential(
            nn.Linear(hidden_size, num_words)
        )

    def forward(self, x, lengths, h=None, c=None):
        x = self.emd(x)

        x = pack_padded_sequence(x, lengths=lengths, batch_first=True, enforce_sorted=False)
        if h is None:
            output, (hn, cn) = self.net(x)
        else:
            output, (hn, cn) = self.net(x, (h, c))
        output = pad_packed_sequence(output, batch_first=True)[0]
        pred = self.classification(output)
        return pred, hn, cn
