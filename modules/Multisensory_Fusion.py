from torch import nn
import torch

class Multisensory_Fusion(nn.Module):
    def __init__(self, args):
        self.args = args
        self.batch_size = args.batch_size
        self.unimodal = unimodal

    def forward(self, r,d,m,t):
        out = torch.Tensor().to(self.args.device_id)
        for i in range(self.batch_size):
            if t is not None:
                tt = t[i].repeat(1, 1, 8, 8)  # (1, 2, 8, 8)
                if self.unimodal:
                    result = tt
            out = torch.cat((out, result), 0)


        return out
