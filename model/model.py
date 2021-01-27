import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from modules.utils import get_hidden_layer_sizes

# https://github.com/curiousily/Getting-Things-Done-with-Pytorch/blob/master/06.time-series-anomaly-detection-ecg.ipynb

class LSTM_AE(nn.Module):

    def __init__(self, args, seq_len, n_features, embedding_dim=64):
        super(LSTM_AE, self).__init__()
        self.args = args

        self.encoder = Encoder(seq_len, args, n_features, embedding_dim).to(args.device_id)
        self.decoder = Decoder(seq_len, args, embedding_dim, n_features).to(args.device_id)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Encoder(nn.Module):
    def __init__(self, seq_len, args, n_features, embedding_dim=16):
        super(Encoder, self).__init__()
        self.args = args
        self.layer_list = []
        use_batch_norm = True

        hidden_sizes = get_hidden_layer_sizes(n_features, embedding_dim, args.n_layer)

        layer_sizes = [n_features] + hidden_sizes + [embedding_dim]
        for idx, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            if idx < len(hidden_sizes):
                layer = nn.LSTM(input_size=in_size,
                            hidden_size=out_size,
                            num_layers=1,
                            batch_first=use_batch_norm)
            else:
                layer = nn.LSTM(input_size=in_size,
                                hidden_size=out_size,
                                num_layers=1,
                                batch_first=use_batch_norm)
            self.layer_list.append(layer)

            self.seq_len, self.n_features = seq_len, n_features
            self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

            self.rnn1 = nn.LSTM(
              input_size=n_features,
              hidden_size=self.hidden_dim,
              num_layers=1,
              batch_first=True
            )

            self.rnn2 = nn.LSTM(
              input_size=self.hidden_dim,
              hidden_size=embedding_dim,
              num_layers=1,
              batch_first=True
            )

    def forward(self, x):
        x = x.reshape((self.args.batch_size, self.seq_len, self.n_features))
        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)
        # return x.reshape((self.args.batch_size, self.seq_len, self.embedding_dim))
        # return hidden_n.reshape((self.n_features, self.embedding_dim))
        return hidden_n.reshape((self.args.batch_size, self.embedding_dim))

class Decoder(nn.Module):

    def __init__(self, seq_len, args, input_dim=16, n_features=1):
        super(Decoder, self).__init__()
        self.args = args
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        self.rnn1 = nn.LSTM(
          input_size=input_dim,
          hidden_size=input_dim,
          num_layers=1,
          batch_first=True
        )

        self.rnn2 = nn.LSTM(
          input_size=input_dim,
          hidden_size=self.hidden_dim,
          num_layers=1,
          batch_first=True
        )

        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        # x = x.repeat(self.seq_len, self.n_features)
        x = x.repeat(self.seq_len, 1)
        x = x.reshape((self.args.batch_size, self.seq_len, self.input_dim))
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        x = x.reshape((self.args.batch_size, self.seq_len, self.hidden_dim))
        return self.output_layer(x)