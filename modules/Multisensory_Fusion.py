from torch import nn
import torch

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

    def fwd(self, r,d,m,t):
        r = r.to(self.args.device_id)
        d = d.to(self.args.device_id)
        m = m.to(self.args.device_id)
        t = t.to(self.args.device_id)
        out = torch.Tensor().to(self.args.device_id)
        for i in range(self.batch_size):
            if t is not None:
                tt = t[i].unsqueeze(1).unsqueeze(1)
                tt = tt.repeat(1, 8, 8)
                tt = tt.unsqueeze(0)
                if self.unimodal:
                    result = tt
            out = torch.cat((out, result), 0)

        out = out.view(-1, 64 * 3)
        return out
