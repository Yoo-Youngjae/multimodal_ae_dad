from torch import nn
import torch
import numpy as np
class Multisensory_Fusion(): # nn.Module
    def __init__(self, args):
        # super(Multisensory_Fusion, self).__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.unimodal = True
        self.All = False
        self.force_torque = False
        if args.sensor == 'All':
            self.All = True
            self.force_torque = True
            self.unimodal = False
        elif args.sensor == 'force_torque':
            self.force_torque = True

        self.conv1m = nn.Conv1d(1, 8, kernel_size=6).to(args.device_id)


    def fwd(self, r,d,m,t):
        # batch normalization
        # t = self.norm_vec(t)
        r = r.to(self.args.device_id)
        d = d.to(self.args.device_id)
        m = m.to(self.args.device_id)
        t = t.to(self.args.device_id)
        out = torch.Tensor().to(self.args.device_id)
        for i in range(self.batch_size):
            if t.shape[1] != 0:     # t is not None
                # t[i].shape == 3
                tt = t[i].unsqueeze(1).unsqueeze(1)
                # tt.shape == 3, 1, 1
                tt = tt.repeat(1, 8, 8)
                # tt.shape == 3, 8, 8
                tt = tt.unsqueeze(0)
                # tt.shape == 1, 3, 8, 8
                if self.unimodal:
                    result = tt
            if m.shape[1] != 0:     # m is not None
                # m[i].shape == 3, 13
                mm = m[i].unsqueeze(1)
                # mm.shape == 3, 1, 13
                mm = self.conv1m(mm)
                # mm.shape == 3, 8, 8
                mm = mm.unsqueeze(0)
                if self.unimodal:
                    result = mm
            if r.shape[1] != 0:
                # r[i].shape == 3,1000
                rr = r[i].unsqueeze(0)
                # rr.shape == 1, 3, 1000
                if self.unimodal:
                    result = rr
            if d.shape[1] != 0:
                # d[i].shape == 3,1000
                dd = d[i].unsqueeze(0)
                # dd.shape == 1, 3, 1000
                if self.unimodal:
                    result = dd
            out = torch.cat((out, result), 0)


        out = out.view(-1, self.args.seq_len, self.args.n_features)
        return out



    def norm_vec(self,v, range_in=None, range_out=None):
        if range_out is None:
            range_out = [0.0,1.0]
        if range_in is None:
            range_in = [torch.min(v, 0)[0], torch.max(v, 0)[0]] #range_in = [np.min(v,0), np.max(v,0)]
        r_out = range_out[1] - range_out[0]
        r_in = range_in[1] - range_in[0]
        v = (r_out * (v - range_in[0]) / r_in) + range_out[0]
        # v = self.nan_to_num(v, nan=0.0)
        return v
