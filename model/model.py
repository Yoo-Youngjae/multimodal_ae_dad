import torch.nn as nn
import torch
from modules.utils import get_hidden_layer_sizes
from modules.Multisensory_Fusion import Multisensory_Fusion


# https://github.com/curiousily/Getting-Things-Done-with-Pytorch/blob/master/06.time-series-anomaly-detection-ecg.ipynb

class LSTM_AE(nn.Module):

    def __init__(self, args, seq_len, n_features, embedding_dim=64):
        super(LSTM_AE, self).__init__()
        self.args = args
        self.multisensory_fusion = Multisensory_Fusion(args)
        self.encoder = Encoder(seq_len, args, n_features, embedding_dim).to(args.device_id)
        self.decoder = Decoder(seq_len, args, embedding_dim, n_features).to(args.device_id)
        self.criterion = nn.MSELoss(reduction='sum').to(args.device_id)

    def fusion(self, r, d, m, t):
        return self.multisensory_fusion(r, d, m, t)

    def forward(self, input_representation):
        input_representation = input_representation.to(self.args.device_id)
        x, encoder_state = self.encoder(input_representation)
        x = self.decoder(x, encoder_state)
        return x
    def get_loss_value(self, x, y):
        output = self(x)
        loss = self.criterion(output, x)
        return loss



class Encoder(nn.Module):
    def __init__(self, seq_len, args, n_features, embedding_dim=16):
        super(Encoder, self).__init__()
        self.args = args
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim

        use_batch_first = True

        hidden_sizes = get_hidden_layer_sizes(n_features, embedding_dim, args.n_layer-1)

        layer_sizes = [n_features] + hidden_sizes + [embedding_dim]

        self.lstm1 = nn.LSTM(input_size=layer_sizes[0],
                            hidden_size=layer_sizes[1],
                            num_layers=1,
                            batch_first=use_batch_first)

        self.lstm2 = nn.LSTM(input_size=layer_sizes[1],
                             hidden_size=layer_sizes[2],
                             num_layers=1,
                             batch_first=use_batch_first)

        self.lstm3 = nn.LSTM(input_size=layer_sizes[2],
                             hidden_size=layer_sizes[3],
                             num_layers=1,
                             batch_first=use_batch_first)

        self.lstm4 = nn.LSTM(input_size=layer_sizes[3],
                             hidden_size=layer_sizes[4],
                             num_layers=1,
                             batch_first=use_batch_first)

        self.lstm5 = nn.LSTM(input_size=layer_sizes[4],
                             hidden_size=layer_sizes[5],
                             num_layers=1,
                             batch_first=use_batch_first)

        # self.lstm = nn.LSTM(input_size=layer_sizes[0],
        #                      hidden_size=layer_sizes[5],
        #                      num_layers=5,
        #                      batch_first=use_batch_norm)

    def forward(self, x):
        x = x.reshape((self.args.batch_size, self.seq_len, self.n_features))

        # multi layer
        x, (_, _) = self.lstm1(x)
        x, (_, _) = self.lstm2(x)
        x, (_, _) = self.lstm3(x)
        x, (_, _) = self.lstm4(x)
        x, encoder_state = self.lstm5(x)

        # single function
        # x, encoder_state = self.lstm(x)

        # x.shape : torch.Size([64, 3, 32])
        # hidden_n.shape : torch.Size([1, 3, 32])
        # self.n_features : 64
        # self.embedding_dim : 32
        return x, encoder_state  # hidden_n

class Decoder(nn.Module):

    def __init__(self, seq_len, args, input_dim=16, n_features=1):
        super(Decoder, self).__init__()
        self.args = args
        self.seq_len = seq_len
        self.n_features = n_features
        use_batch_first = True

        hidden_sizes = get_hidden_layer_sizes(input_dim, n_features, args.n_layer-1)
        layer_sizes = [input_dim] + hidden_sizes + [n_features]
        self.input_dim = input_dim
        self.hidden_dim = layer_sizes[4]

        self.lstm1 = nn.LSTM(input_size=layer_sizes[0],
                             hidden_size=layer_sizes[0],
                             num_layers=1,
                             batch_first=use_batch_first)

        self.lstm2 = nn.LSTM(input_size=layer_sizes[0],
                             hidden_size=layer_sizes[1],
                             num_layers=1,
                             batch_first=use_batch_first)

        self.lstm3 = nn.LSTM(input_size=layer_sizes[1],
                             hidden_size=layer_sizes[2],
                             num_layers=1,
                             batch_first=use_batch_first)

        self.lstm4 = nn.LSTM(input_size=layer_sizes[2],
                             hidden_size=layer_sizes[3],
                             num_layers=1,
                             batch_first=use_batch_first)

        self.lstm5 = nn.LSTM(input_size=layer_sizes[3],
                             hidden_size=layer_sizes[4],
                             num_layers=1,
                             batch_first=use_batch_first)

        # self.lstm = nn.LSTM(input_size=layer_sizes[0],
        #                      hidden_size=layer_sizes[4],
        #                      num_layers=5,
        #                      batch_first=use_batch_norm)

        self.output_layer = nn.Linear(layer_sizes[4], layer_sizes[5])



    def forward(self, x, encoder_state):

        # x = x.repeat(self.args.batch_size, 1,  1)
        x = x.reshape((self.args.batch_size, self.seq_len, self.input_dim))

        # Multi layer
        x, (hidden_n, _) = self.lstm1(x, encoder_state) # encoder_state
        x, (hidden_n, _) = self.lstm2(x)
        x, (hidden_n, _) = self.lstm3(x)
        x, (hidden_n, _) = self.lstm4(x)
        x, (hidden_n, _) = self.lstm5(x)


        # Single function
        # x, (hidden_n, _) = self.lstm(x)

        x = x.reshape((self.args.batch_size, self.seq_len, self.hidden_dim))
        return self.output_layer(x)





class DNN_AE(nn.Module):

    def __init__(self, args, seq_len, n_features, embedding_dim=64):
        super(DNN_AE, self).__init__()
        self.args = args
        self.multisensory_fusion = Multisensory_Fusion(args)
        # n_features = n_features * 3
        self.encoder = DNN_Encoder(seq_len, args, n_features, embedding_dim).to(args.device_id)
        self.decoder = DNN_Decoder(seq_len, args, embedding_dim, n_features).to(args.device_id)
        self.criterion = nn.MSELoss(reduction='sum').to(args.device_id)

    def fusion(self, r, d, m, t):
        return self.multisensory_fusion(r, d, m, t)

    def forward(self, input_representation):
        input_representation = input_representation.to(self.args.device_id)
        x = self.encoder(input_representation)
        x = self.decoder(x)
        return x #, input_representation.reshape((self.args.batch_size, self.seq_len, self.n_features)) #((self.args.batch_size, -1))

    def get_loss_value(self, x, y):
        output = self(x)
        loss = self.criterion(output, x)
        return loss



class DNN_Encoder(nn.Module):
    def __init__(self, seq_len, args, n_features, embedding_dim=16):
        super(DNN_Encoder, self).__init__()
        self.args = args
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim

        use_batch_norm = False

        hidden_sizes = get_hidden_layer_sizes(n_features, embedding_dim, args.n_layer-1)
        self.layer_list = []
        layer_sizes = [n_features] + hidden_sizes + [embedding_dim]
        for idx, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            if idx < len(hidden_sizes):
                layer = FCLayer(input_size=in_size,
                                output_size=out_size,
                                act='leakyrelu',
                                bn=use_batch_norm,
                                dropout_p=0
                                )
            else:
                layer = FCLayer(input_size=in_size,
                                output_size=out_size,
                                act=None,
                                )
            self.layer_list.append(layer)
        self.net = nn.Sequential(*self.layer_list)

    def forward(self, x):
        x = x.reshape((self.args.batch_size, self.seq_len, self.n_features))
        # x = x.reshape((self.args.batch_size, -1))
        return self.net(x)

class DNN_Decoder(nn.Module):

    def __init__(self, seq_len, args, input_dim=16, n_features=1):
        super(DNN_Decoder, self).__init__()
        self.args = args
        self.seq_len = seq_len
        self.n_features = n_features
        self.input_dim = input_dim
        use_batch_norm = False

        hidden_sizes = get_hidden_layer_sizes(n_features, n_features, args.n_layer - 1)
        self.layer_list = []
        layer_sizes = [input_dim] + hidden_sizes + [n_features]
        for idx, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            if idx < len(hidden_sizes):
                layer = FCLayer(input_size=in_size,
                                output_size=out_size,
                                act='leakyrelu',
                                bn=use_batch_norm,
                                dropout_p=0
                                )
            else:
                layer = FCLayer(input_size=in_size,
                                output_size=out_size,
                                act=None,
                                )
            self.layer_list.append(layer)
        self.net = nn.Sequential(*self.layer_list)

    def forward(self, x):
        x = x.reshape((self.args.batch_size, self.seq_len, self.input_dim))
        # x = x.reshape((self.args.batch_size, -1))
        return self.net(x)


class FCLayer(nn.Module):
    def __init__(self,
                 input_size,
                 output_size=1,
                 bias=True,
                 act='relu',
                 bn=False,
                 dropout_p=0):
        super().__init__()
        self.layer = nn.Linear(input_size, output_size, bias)
        self.bn = nn.BatchNorm1d(output_size) if bn else None
        self.dropout = nn.Dropout(dropout_p) if dropout_p else None
        self.act = Activation(act) if act else None

    def forward(self, x):
        y = self.act(self.layer(x)) if self.act else self.layer(x)
        if self.bn:
            # In case of expansion(k) in Information bottleneck
            if y.dim() > 2:
                original_y_size = y.size()
                y = self.bn(y.view(-1, y.size(-1))).view(*original_y_size)
            else:
                y = self.bn(y)
        y = self.dropout(y) if self.dropout else y

        return y


class Activation(nn.Module):

    def __init__(self, act):
        super().__init__()

        if act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'logsigmoid':
            self.act = nn.LogSigmoid()
        elif act == 'softmax':
            self.act = nn.Softmax(dim=-1)
        elif act == 'logsoftmax':
            self.act = nn.LogSoftmax(dim=-1)
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'relu':
            self.act = nn.ReLU()
        elif act == 'leakyrelu':
            self.act = nn.LeakyReLU(.2)
        else:
            self.act = None

    def forward(self, x):
        if self.act is not None:
            return self.act(x)
        return x
