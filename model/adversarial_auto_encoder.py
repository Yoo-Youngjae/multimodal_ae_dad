#
#  MAKINAROCKS CONFIDENTIAL
#  ________________________
#
#  [2017] - [2020] MakinaRocks Co., Ltd.
#  All Rights Reserved.
#
#  NOTICE:  All information contained herein is, and remains
#  the property of MakinaRocks Co., Ltd. and its suppliers, if any.
#  The intellectual and technical concepts contained herein are
#  proprietary to MakinaRocks Co., Ltd. and its suppliers and may be
#  covered by U.S. and Foreign Patents, patents in process, and
#  are protected by trade secret or copyright law. Dissemination
#  of this information or reproduction of this material is
#  strictly forbidden unless prior written permission is obtained
#  from MakinaRocks Co., Ltd.
import torch

from model.abstract_model import AbstractModel
from modules.fc_module import FCModule
from modules.utils import get_hidden_layer_sizes, Loss
from modules.Multisensory_Fusion import Multisensory_Fusion


class AdversarialAutoEncoder(AbstractModel):
    def __init__(
        self,
        args,
        input_size,
        n_layers=10,
        btl_size=100,
        encoder_and_opt=None,
        decoder_and_opt=None,
        discriminator_and_opt=None,
        lr=0.001,
        recon_loss=Loss('mse', reduction="sum"),
        bce_loss=Loss('bce'),
    ):
        super().__init__()
        self.args = args
        self.btl_size = btl_size
        default_opt = torch.optim.Adam
        self.multisensory_fusion = Multisensory_Fusion(args)
        if encoder_and_opt is None:
            default_encoder = FCModule(
                input_size=input_size,
                output_size=btl_size,
                hidden_sizes=get_hidden_layer_sizes(input_size, btl_size, n_hidden_layers=n_layers - 1),
                use_batch_norm=True,
                act="leakyrelu",
                last_act=None,
            )
            self.encoder = default_encoder
            self.encoder_opt = default_opt(self.encoder.parameters(), lr=lr)
        else:
            self.encoder = encoder_and_opt[0]
            self.encoder_opt = encoder_and_opt[1]

        if decoder_and_opt is None:
            default_decoder = FCModule(
                input_size=btl_size,
                output_size=input_size,
                hidden_sizes=get_hidden_layer_sizes(btl_size, input_size, n_hidden_layers=n_layers - 1),
                use_batch_norm=True,
                act="leakyrelu",
                last_act=None,
            )
            self.decoder = default_decoder
            self.decoder_opt = default_opt(self.decoder.parameters(), lr=lr)
        else:
            self.decoder = decoder_and_opt[0]
            self.decoder_opt = decoder_and_opt[1]

        if discriminator_and_opt is None:
            default_discriminator = FCModule(
                input_size=btl_size,
                output_size=1,
                hidden_sizes=get_hidden_layer_sizes(btl_size, 1, n_hidden_layers=n_layers - 1),
                use_batch_norm=False,
                dropout_p=0.2,
                act="relu",
                last_act="sigmoid"
            )
            self.discriminator = default_discriminator
            self.discriminator_opt = default_opt(self.discriminator.parameters(), lr=lr / 2)
        else:
            self.discriminator = discriminator_and_opt[0]
            self.discriminator_opt = discriminator_and_opt[1]

        self.recon_loss = recon_loss
        self.bce_loss = bce_loss

    def forward(self, x):
        x = x.to(self.args.device_id)
        return self.decoder(self.encoder(x))

    def fusion(self, r, d, m, t):
        return self.multisensory_fusion(r, d, m, t)

    def get_recon_loss(self, x, y, *args, **kwargs):
        x_recon = self.forward(x)
        recon_loss = self.recon_loss(x_recon, x)
        return recon_loss

    def get_D_loss(self, x, y, *args, **kwargs):
        device = x.device

        z_fake = self.encoder(x)
        z_true = torch.randn(x.size(0), self.btl_size).to(device)

        z_true_pred = self.discriminator(z_true)
        z_fake_pred = self.discriminator(z_fake)

        target_ones = torch.ones(x.size(0), 1).to(device)
        target_zeros = torch.zeros(x.size(0), 1).to(device) 

        true_loss = self.bce_loss(z_true_pred, target_ones)
        fake_loss = self.bce_loss(z_fake_pred, target_zeros)

        D_loss = true_loss + fake_loss
        return D_loss

    def get_G_loss(self, x, y, *args, **kwargs):
        target_ones = torch.ones(x.size(0), 1).to(x.device)
        z_fake = self.encoder(x)
        z_fake_pred = self.discriminator(z_fake)
        G_loss = self.bce_loss(z_fake_pred, target_ones)
        return G_loss


    def step(self, x):
        self.train()

        x = x.cuda(self.args.device_id)

        # update: decoder, encoder
        self.decoder_opt.zero_grad()
        self.encoder_opt.zero_grad()
        recon_loss = self.get_recon_loss(x, None)
        recon_loss.backward(retain_graph=True)
        self.decoder_opt.step()
        self.encoder_opt.step()

        # update: discriminator
        self.discriminator_opt.zero_grad()
        D_loss = self.get_D_loss(x, None)
        D_loss.backward(retain_graph=True)
        self.discriminator_opt.step()

        # update: generator
        self.encoder_opt.zero_grad()
        G_loss = self.get_G_loss(x, None)
        G_loss.backward(retain_graph=True)
        self.encoder_opt.step()

        return float(recon_loss) #, float(D_loss), float(G_loss)
    #
    # @staticmethod
    # def validate(engine, mini_batch):
    #     # idx_loss = 0
    #     engine.model.eval()
    #
    #     with torch.no_grad():
    #         x, _ = mini_batch
    #         if engine.config.gpu_id >= 0:
    #             x = x.cuda(engine.config.gpu_id)
    #         recon_loss = engine.model.get_recon_loss(x, None)
    #         D_loss = engine.model.get_D_loss(x, None)
    #         G_loss = engine.model.get_G_loss(x, None)
    #
    #     return float(recon_loss), float(D_loss), float(G_loss)

