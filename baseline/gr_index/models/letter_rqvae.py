import sys
import os
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../index'))

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rqvae import RQVAE
from models.vq import VectorQuantizer
from models.layers import MLPLayers
from baseline.gr_index.models.lcrec_rqvae import BalancedSinkhornVQ, BalancedSinkhornRQ


class LetterRQVAE(RQVAE):

    def __init__(self, orth_loss_weight=0.1, **kwargs):
        num_layers = len(kwargs.get('num_emb_list', [256, 256, 256, 256]))
        kwargs['sk_epsilons'] = [0.003] * num_layers
        super().__init__(**kwargs)
        self.use_ot_loss = False
        self.ot_loss_weight = 0.0
        self.orth_loss_weight = orth_loss_weight

        self.rq = BalancedSinkhornRQ(
            kwargs['num_emb_list'], kwargs['e_dim'],
            kwargs['sk_epsilons'],
            kmeans_init=kwargs.get('kmeans_init', True),
            kmeans_iters=kwargs.get('kmeans_iters', 100),
            sk_iters=kwargs.get('sk_iters', 100),
            use_linear=kwargs.get('use_linear', 0),
        )

    def forward(self, x, use_sk=True):
        x = self.encoder(x)
        x_q, rq_loss, indices, distances = self.rq(x, use_sk=use_sk)
        out = self.decoder(x_q)
        return out, rq_loss, indices

    def compute_loss(self, out, quant_loss, xs=None, orth_loss=None):
        if self.loss_type == 'mse':
            loss_recon_point = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon_point = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')

        loss_recon = loss_recon_point

        orth_loss_val = 0.0
        if self.orth_loss_weight > 0:
            codebook = self.rq.get_codebook()
            for cb in codebook:
                cb_norm = F.normalize(cb, dim=1)
                gram = torch.mm(cb_norm, cb_norm.T)
                identity = torch.eye(gram.shape[0], device=gram.device)
                orth_loss_val += F.mse_loss(gram, identity)

        loss_total = 1.2 * loss_recon + self.quant_loss_weight * quant_loss + \
                     self.orth_loss_weight * orth_loss_val

        return loss_total, loss_recon

    @torch.no_grad()
    def get_indices(self, xs, use_sk=False):
        x_e = self.encoder(xs)
        _, _, indices, distances = self.rq(x_e, use_sk=use_sk)
        return indices, distances
