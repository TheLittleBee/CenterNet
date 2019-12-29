from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from models.losses import FocalLoss, MSELoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.utils import _sigmoid
from .base_trainer import BaseTrainer


class TTFLoss(torch.nn.Module):
    def __init__(self, opt):
        super(TTFLoss, self).__init__()
        self.crit = MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.opt = opt

    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss, wh_loss = 0, 0
        for s in range(opt.num_stacks):
            hm, wh = outputs[s]
            if not opt.mse_loss:
                hm = _sigmoid(hm)

            hm_loss += self.crit(hm, batch['hm']) / opt.num_stacks
            if opt.wh_weight > 0:
                twh = batch['wh'] / 2
                target = torch.cat([twh - batch['reg'], twh + batch['reg']], dim=2)
                wh_loss += self.crit_reg(
                    wh, batch['reg_mask'],
                    batch['ind'], target) / opt.num_stacks

        loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss
        loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                      'wh_loss': wh_loss}
        return loss, loss_stats


class TTFTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(TTFTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss']
        loss = TTFLoss(opt)
        return loss_states, loss

    def debug(self, batch, output, iter_id):
        raise NotImplementedError

    def save_result(self, output, batch, results):
        raise NotImplementedError
