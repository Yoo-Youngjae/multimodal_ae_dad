from torch import nn
import torch
import torch.nn.functional as F
from torchvision.transforms import Normalize

class Multisensory_Fusion(nn.Module): # nn.Module
    def __init__(self, args):
        super(Multisensory_Fusion, self).__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.unimodal = True
        if args.sensor == 'All':
            self.unimodal = False

        # for Multimodal
        self.layer1 = nn.Sequential(
            # 4 x 32 x 32
            nn.Conv2d(4, 64, kernel_size=2, stride=2),
            # 64 x 16 x 16
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            # 32 x 16 x 16
            nn.LeakyReLU(),
            nn.BatchNorm2d(32)
        ).to(self.args.device_id)

        self.layer2 = nn.Sequential(
            # 34 x 16 x 16
            nn.Conv2d(34, 32, kernel_size=3, stride=1, padding=1),
            # 32 x 16 x 16
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
        ).to(self.args.device_id)

        self.layer3 = nn.Sequential(
            # 32 x 16 x 16
            nn.Conv2d(32, 32, kernel_size=2, stride=2),
            # 32 x 8 x 8
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
        ).to(self.args.device_id)

        # for Unimodal
        # self.conv1r = nn.Conv2d(3, 8, kernel_size=2, stride=2).to(self.args.device_id)
        # self.conv2r = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1).to(self.args.device_id)
        # self.conv3r = nn.Conv2d(8, 8, kernel_size=2, stride=2).to(self.args.device_id),


    def forward(self, r, d, m, t):
        # batch normalization
        if self.args.sensor == 'All':
            # r = self.norm_vec(r, range_in=[0, 255])
            # d = self.norm_vec(d, range_in=[0, 255])
            # m = self.norm_vec(m)
            # t = self.norm_vec(t)
            im = torch.cat((r, d), 2)
            im = im.to(self.args.device_id)
        elif self.args.sensor == 'force_torque':
            t = self.norm_vec(t)

        # r[i] :   torch.Size([3, 3, 32, 32])
        # d[i] :   torch.Size([3, 1, 32, 32])
        # m[i] :   torch.Size([3, 13])
        # t[i] :   torch.Size([3])
        # im[i] :   torch.Size([3, 4, 32, 32])
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
                tt = tt.repeat(1, 16, 16)
                # tt.shape == 3, 16, 16
                tt = tt.unsqueeze(1)
                # tt.shape == 3, 1, 16, 16

                if self.unimodal:
                    result = tt
            if m.shape[1] != 0:     # m is not None
                # m[i].shape == 3, 16
                mm = m[i].unsqueeze(1)
                # mm.shape == 3, 1, 16
                mm = mm.repeat(1, 16, 1)
                # mm.shape == 3, 16, 16
                mm = mm.unsqueeze(1)
                # mm.shape == 3, 1, 16, 16

                if self.unimodal:
                    result = mm
            if self.args.sensor == 'hand_camera':   # unimodal
                # r[i].shape == 3, 3, 32, 32
                rr = self.conv1r(r[i])
                # rr.shape == 3, 8, 16, 16
                rr = self.conv2r(rr)
                # im.shape == 3, 8, 16, 16
                rr = self.conv3r(rr)
                # im.shape == 3, 8, 8, 8
                result = rr.unsqueeze(0)
                # result.shape == 1, 3, 8, 8, 8

            if self.args.sensor == 'All':
                # im[i].shape == 3, 4, 32, 32
                # batch norm
                imim = self.layer1(im[i])
                # imim.shape == 3, 32, 16, 16
                multimodal = torch.cat((imim, tt, mm), 1)
                # multimodal.shape == 3, 34, 16, 16
                multimodal = self.layer2(multimodal) + imim
                # multimodal.shape == 3, 32, 16, 16
                multimodal = self.layer3(multimodal)
                # multimodal.shape == 3, 32, 8, 8
                result = multimodal.unsqueeze(0)
                # result.shape == 1, 3, 32, 8, 8
            out = torch.cat((out, result), 0)

        out = out.view(self.batch_size, self.args.seq_len, self.args.n_features)
        return out

    ## todo : Normalization!!!
    def norm_vec(self,v, range_in=None, range_out=None):
        if range_out is None:
            range_out = [-1, 1]
        if range_in is None:
            range_in = [torch.min(v), torch.max(v)]

        r_out = range_out[1] - range_out[0]
        r_in = range_in[1] - range_in[0]
        v = (r_out * (v - range_in[0]) / r_in) + range_out[0]
        return v
