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
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim

        use_batch_norm = False

        hidden_sizes = get_hidden_layer_sizes(n_features, embedding_dim, args.n_layer-1)

        layer_sizes = [n_features] + hidden_sizes + [embedding_dim]

        self.lstm1 = nn.LSTM(input_size=layer_sizes[0],
                            hidden_size=layer_sizes[1],
                            num_layers=1,
                            batch_first=use_batch_norm)

        self.lstm2 = nn.LSTM(input_size=layer_sizes[1],
                             hidden_size=layer_sizes[2],
                             num_layers=1,
                             batch_first=use_batch_norm)

        self.lstm3 = nn.LSTM(input_size=layer_sizes[2],
                             hidden_size=layer_sizes[3],
                             num_layers=1,
                             batch_first=use_batch_norm)

        self.lstm4 = nn.LSTM(input_size=layer_sizes[3],
                             hidden_size=layer_sizes[4],
                             num_layers=1,
                             batch_first=use_batch_norm)

        self.lstm5 = nn.LSTM(input_size=layer_sizes[4],
                             hidden_size=layer_sizes[5],
                             num_layers=1,
                             batch_first=use_batch_norm)




    def forward(self, x):
        x = x.reshape((self.args.batch_size, self.seq_len, self.n_features))
        x, (_, _) = self.lstm1(x)
        x, (_, _) = self.lstm2(x)
        x, (_, _) = self.lstm3(x)
        x, (_, _) = self.lstm4(x)
        x, (hidden_n, _) = self.lstm5(x)
        # print('x',x.shape)
        # print('self.args.batch_size', self.args.batch_size, 'self.embedding_dim', self.embedding_dim)
        return x # x.reshape((self.args.batch_size, self.embedding_dim)) # hidden_n

class Decoder(nn.Module):

    def __init__(self, seq_len, args, input_dim=16, n_features=1):
        super(Decoder, self).__init__()
        self.args = args
        self.seq_len = seq_len
        self.n_features = n_features
        use_batch_norm = False

        hidden_sizes = get_hidden_layer_sizes(input_dim, n_features, args.n_layer-1)
        layer_sizes = [input_dim] + hidden_sizes + [n_features]
        self.input_dim = input_dim
        self.hidden_dim =  layer_sizes[4]

        self.lstm1 = nn.LSTM(input_size=layer_sizes[0],
                             hidden_size=layer_sizes[0],
                             num_layers=1,
                             batch_first=use_batch_norm)

        self.lstm2 = nn.LSTM(input_size=layer_sizes[0],
                             hidden_size=layer_sizes[1],
                             num_layers=1,
                             batch_first=use_batch_norm)

        self.lstm3 = nn.LSTM(input_size=layer_sizes[1],
                             hidden_size=layer_sizes[2],
                             num_layers=1,
                             batch_first=use_batch_norm)

        self.lstm4 = nn.LSTM(input_size=layer_sizes[2],
                             hidden_size=layer_sizes[3],
                             num_layers=1,
                             batch_first=use_batch_norm)

        self.lstm5 = nn.LSTM(input_size=layer_sizes[3],
                             hidden_size=layer_sizes[4],
                             num_layers=1,
                             batch_first=use_batch_norm)

        self.output_layer = nn.Linear(layer_sizes[4], layer_sizes[5])



    def forward(self, x):
        # x = x.repeat(self.seq_len, self.n_features)
        # x = x.repeat(self.seq_len, 1)
        # x = x.reshape((self.args.batch_size, self.seq_len, self.input_dim))
        x, (hidden_n, _) = self.lstm1(x)
        x, (hidden_n, _) = self.lstm2(x)
        x, (hidden_n, _) = self.lstm3(x)
        x, (hidden_n, _) = self.lstm4(x)
        x, (hidden_n, _) = self.lstm5(x)
        x = x.reshape((self.args.batch_size, self.seq_len, self.hidden_dim))
        return self.output_layer(x)