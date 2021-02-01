from torch import nn
import torch
import numpy as np
class Multisensory_Fusion(): # nn.Module
    def __init__(self, args):
        # super(Multisensory_Fusion, self).__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.unimodal = True
        if args.sensor == 'All':
            self.unimodal = False


        self.conv1im = nn.Conv2d(4, 8, kernel_size=2, stride=2)
        self.conv2im = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        self.conv3im = nn.Conv2d(8, 8, kernel_size=2, stride=2)

    def fwd(self, r, d, m, t):
        # batch normalization
        # t = self.norm_vec(t)
        im = torch.cat((r, d), 2)

        # r[i] :   torch.Size([3, 3, 32, 32])
        # d[i] :   torch.Size([3, 1, 32, 32])
        # m[i] :   torch.Size([3, 13])
        # t[i] :   torch.Size([3])
        # im[i] :   torch.Size([3, 4, 32, 32])

        print(r.shape, d.shape, m.shape, t.shape)
        im = im.to(self.args.device_id)
        m = m.to(self.args.device_id)
        t = t.to(self.args.device_id)
        out = torch.Tensor().to(self.args.device_id)

        for i in range(self.batch_size):
            if t.shape[1] != 0:     # t is not None
                # t[i].shape == 3
                tt = t[i].unsqueeze(1).unsqueeze(1)
                # tt.shape == 3, 1, 1
                tt = tt.repeat(1, 16, 16)
                # tt.shape == 3, 16, 16
                tt = tt.unsqueeze(1)
                # tt.shape == 3, 1, 16, 16

                if self.unimodal:
                    result = tt
            if m.shape[1] != 0:     # m is not None
                # m[i].shape == 3, 13
                mm = m[i].unsqueeze(1)
                # mm.shape == 3, 1, 13
                # todo : 16 -> (16, 16)
                mm = mm.repeat(1, 16, 1)
                # mm.shape == 3, 16, 16
                mm = mm.unsqueeze(1)
                # mm.shape == 3, 1, 16, 16

                if self.unimodal:
                    result = mm
            if im.shape[1] != 0:    # im is not None
                # im[i].shape == 3, 4, 32, 32
                imim = self.conv1im(im[i])
                # imim.shape == 3, 8, 16, 16

                if self.unimodal:
                    result = imim

            if not self.unimodal:
                im = torch.cat((imim, tt,mm), 1)
                # im.shape == 3, 10, 16, 16
                result = im.unsqueeze(0)
            out = torch.cat((out, result), 0)


        out = out.view(-1, self.args.seq_len, self.args.n_features)
        return out

    ## todo : Normalization!!!
    def norm_vec(self,v, range_in=None, range_out=None):
        if range_out is None:
            range_out = [-1, 1]
        if range_in is None:
            range_in = [torch.min(v, 0), torch.max(v, 0)] # [0] [0]
        r_out = range_out[1] - range_out[0]
        r_in = range_in[1] - range_in[0]
        v = (r_out * (v - range_in[0]) / r_in) + range_out[0]
        # v = self.nan_to_num(v, nan=0.0)
        return v
