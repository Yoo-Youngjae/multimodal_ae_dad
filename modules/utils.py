import numpy as np
import torch
import torch.nn as nn
def get_hidden_layer_sizes(start_size, end_size, n_hidden_layers):
    """
    It can handle both increasing & decreasing sizes automatically
    """
    sizes = []
    diff = (start_size - end_size) / (n_hidden_layers + 1)

    for idx in range(n_hidden_layers):
        sizes.append(int(start_size - (diff * (idx + 1))))
    return sizes


def get_diffs(args, x, model):
    model.eval()
    batch_size = args.batch_size
    if isinstance(x, np.ndarray):
        x = torch.tensor(x)
    # x.shape == 234, 128, 3, 2048

    if args.ae_type == 'lstm':
        layerlist =[model.encoder.lstm1, model.encoder.lstm2, model.encoder.lstm3, model.encoder.lstm4, model.encoder.lstm5]
    else:
        layerlist = model.encoder.layer_list

    stacked = []
    for _x in x:
        # _x.shape == 128, 3, 2048
        model.eval()
        _diffs = []
        _x = _x.to(next(model.parameters()).device).float()
        x_tilde = model(_x)
        _diffs.append((x_tilde - _x).reshape(batch_size, -1).cpu())

        for layer in layerlist: #[model.encoder.lstm1, model.encoder.lstm2, model.encoder.lstm3, model.encoder.lstm4, model.encoder.lstm5]:
            _x = layer(_x)[0]               # x, state
            x_tilde = layer(x_tilde)[0]     # x, state
            _diffs.append((x_tilde - _x).reshape(batch_size, -1).cpu())

        # diffs.shape == 6, 128, 2048 * 3
        stacked.append(_diffs)
    # stacked.shape ==
    stacked = list(zip(*stacked))
    # stacked.shape ==
    diffs = [torch.cat(s, dim=0).numpy() for s in stacked]


    return diffs


class Standardizer():

    def __init__(self, *args, **kwargs):
        self.mu, self.var = None, None

    def fit(self, x):
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            x = x.float()
            # |x| = (batch_size, dim)

            self.mu = x.mean(dim=0)
            x = x - self.mu
            self.var = torch.from_numpy(np.cov(x.numpy().T)).diagonal().float()

    def run(self, x):
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            x = x.float()
            # |x| = (batch_size, dim)

            x = (x - self.mu) / self.var**.5

        return x.numpy()

class Rotater():

    def __init__(self, *args, **kwargs):
        self.mu, self.v = None, None

    def fit(self, x, gpu_id=1):
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            x = x.float()
            # |x| = (batch_size, dim)

            self.mu = x.mean(dim=0)
            x = x - self.mu

            device = torch.device('cuda:%d' % gpu_id)

            x = x.to(device)

            u, s, self.v = x.svd()

            if gpu_id >= 0:
                self.v = self.v.cpu()

    def run(self, x, gpu_id=1, max_size=20000):
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            x = x.float()
            # |x| = (batch_size, dim)

            x = x - self.mu
            device = torch.device('cpu') if gpu_id < 0 else torch.device('cuda:%d' % gpu_id)

            x = x.to(device)
            v = self.v.to(device)


            if len(x) > max_size:
                x_tilde = []
                for x_i in x.split(max_size, dim=0):
                    x_i_tilde = torch.matmul(x_i, v)

                    x_tilde += [x_i_tilde]

                x_tilde = torch.cat(x_tilde, dim=0)
            else:
                x_tilde = torch.matmul(x, v)


            x_tilde = x_tilde.cpu()

            return x_tilde.numpy()

class Truncater(Rotater):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, x, trunc, gpu_id=-1, max_size=20000):
        if trunc <= 0:
            return x

        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            x = x.float()
            # |x| = (batch_size, dim)

            x = x - self.mu

            if gpu_id >= 0:
                device = torch.device('cpu') if gpu_id < 0 else torch.device('cuda:%d' % gpu_id)

                x = x.to(device)
                v = self.v.to(device)[:, :trunc]
            else:
                v = self.v[:, :trunc]
            v_t = v.transpose(0, 1)

            if len(x) > max_size:
                x_tilde = []
                for x_i in x.split(max_size, dim=0):
                    x_i_tilde = torch.matmul(torch.matmul(x_i, v), v_t)

                    x_tilde += [x_i_tilde]

                x_tilde = torch.cat(x_tilde, dim=0)
            else:
                x_tilde = torch.matmul(torch.matmul(x, v), v_t)

            if gpu_id >= 0:
                x_tilde = x_tilde.cpu()
            x_tilde = x_tilde + self.mu

            return x_tilde.numpy()

class Loss(nn.Module):

    def __init__(self, loss, weight=None, reduction='sum'):
        self.reduction = reduction

        super().__init__()

        if loss == 'bce':
            self.loss = nn.BCELoss(weight=weight, reduction=reduction)
        elif loss == 'bce_with_logit':
            self.loss = nn.BCEWithLogitsLoss(weight=weight, reduction=reduction)
        elif loss == 'mse':
            self.loss = nn.MSELoss(reduction=reduction)
        elif loss == 'l1':
            self.loss = nn.L1Loss(reduction=reduction)
        elif loss == 'ce':
            self.loss = nn.CrossEntropyLoss(weight=weight, reduction=reduction)
        elif loss == 'nll':
            self.loss = nn.NLLLoss(weight=weight, reduction=reduction)
        else:
            self.loss = None

    def is_classification_task(self):
        if isinstance(self.loss, nn.NLLLoss) or isinstance(self.loss, nn.CrossEntropyLoss):
            return True
        return False

    def forward(self, y_hat, y):
        if self.loss is not None:
            if self.is_classification_task():
                y = y.long()

            return self.loss(y_hat, y)
        return y_hat.mean()