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


class VariationalAutoEncoder(AbstractModel):

    def __init__(
            self,
            args,
            input_size,
            btl_size,
            n_layers=10,
            encoder_and_opt=None,
            decoder_and_opt=None,
            recon_loss=Loss("mse", reduction="sum"),
            kl_loss=lambda mu, logvar: 0.5 * (mu**2 + logvar.exp() - logvar - 1).sum(),
            # For vib
            stochastic_inference=True,
            distribution="normal",
            k=10,
    ):
        super().__init__(encoder_and_opt, decoder_and_opt)
        self.args = args
        self.multisensory_fusion = Multisensory_Fusion(args)
        self.input_size = input_size
        self.btl_size = btl_size
        default_opt = torch.optim.Adam

        if encoder_and_opt is None:
            default_encoder = FCModule(
                input_size=input_size,
                output_size=btl_size * 2,
                hidden_sizes=get_hidden_layer_sizes(input_size, btl_size, n_hidden_layers=n_layers - 1),
                use_batch_norm=True,
                act="leakyrelu",
                last_act=None,
            )
            self.encoder = default_encoder
            self.opt1 = default_opt(self.encoder.parameters())
        else:
            self.encoder = encoder_and_opt[0]
            self.opt1 = encoder_and_opt[1]
        self.optimizer_list.append(self.opt1)

        if decoder_and_opt is None:
            default_decoder = FCModule(
                input_size=btl_size,
                output_size=input_size,
                hidden_sizes=get_hidden_layer_sizes(btl_size, input_size, n_hidden_layers=n_layers - 1),
                use_batch_norm=True,
                act="leakyrelu",
                last_act=None
            )
            self.decoder = default_decoder
            self.opt2 = default_opt(self.decoder.parameters())
        else:
            self.decoder = decoder_and_opt[0]
            self.opt2 = decoder_and_opt[1]
        self.optimizer_list.append(self.opt2)

        self.recon_loss = recon_loss
        self.kl_loss = kl_loss

        self.stochastic_inference = stochastic_inference
        self.distribution = distribution
        self.k = k

    def forward(self, x):
        """
        Model specific logic
        """
        x = x.to(self.args.device_id)
        z_dict = self.encoder(x, distribution=self.distribution, k=self.k,
                              stochastic_inference=self.stochastic_inference)
        x_hat = self.decoder(z_dict['z'])
        return x_hat.mean(dim=0)

    def fusion(self, r, d, m, t):
        return self.multisensory_fusion(r, d, m, t)

    def get_loss_value(self, x, y, *args, **kwargs):
        z_dict = self.encoder(x, distribution=self.distribution, k=self.k,
                              stochastic_inference=self.stochastic_inference)
        mu = z_dict['mu']
        logvar = z_dict['logvar']

        y_hat = self.decoder(z_dict['z'])
        if y_hat.size() == y.size():
            raise NotImplementedError
        else:
            y_hat = y_hat.view(-1, *(y_hat.size()[2:]))

            x_ = x.unsqueeze(0).expand(self.k, *x.size()).contiguous().view(-1, *(x.size()[1:]))

            loss = self.recon_loss(y_hat, x_)
            if self.recon_loss.reduction == 'sum':
                loss = loss / self.k
            else:
                loss = loss * (self.input_size / self.btl_size)

        return loss, self.kl_loss(mu, logvar)

    # @staticmethod
    # def step(engine, mini_batch):
    #     engine.model.train()
    #     engine.optimizer.zero_grad()
    #
    #     x, _ = mini_batch
    #     if engine.config.gpu_id >= 0:
    #         x = x.cuda(engine.config.gpu_id)
    #     x = x.view(x.size(0), -1)
    #
    #     recon_err, kld = engine.model.get_loss_value(x, x)
    #     loss = recon_err + kld
    #
    #     loss.backward(retain_graph=True)
    #
    #     engine.optimizer.step()
    #
    #     return float(loss), float(recon_err), float(kld)
    #
    # @staticmethod
    # def validate(engine, mini_batch):
    #     engine.model.eval()
    #
    #     with torch.no_grad():
    #         x, _ = mini_batch
    #         if engine.config.gpu_id >= 0:
    #             x = x.cuda(engine.config.gpu_id)
    #         x = x.view(x.size(0), -1)
    #
    #         recon_err, kld = engine.model.get_loss_value(x, x)
    #         loss = recon_err + kld
    #
    #     return float(loss), float(recon_err), float(kld)
