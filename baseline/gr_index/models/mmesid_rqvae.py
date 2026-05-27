import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../index'))

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rqvae import RQVAE
from models.layers import MLPLayers
from baseline.gr_index.models.lcrec_rqvae import BalancedSinkhornRQ


class MMESIDRQVAE(nn.Module):

    def __init__(self, text_in_dim=4096, image_in_dim=768,
                 num_emb_list=None, e_dim=64, layers=None,
                 dropout_prob=0.0, bn=False, loss_type='mse',
                 quant_loss_weight=1.0, kmeans_init=True,
                 kmeans_iters=100, sk_iters=100, use_linear=0,
                 cross_modal_weight=0.1):
        super().__init__()
        num_layers = len(num_emb_list) if num_emb_list else 4
        sk_epsilons = [0.003] * num_layers

        self.text_in_dim = text_in_dim
        self.image_in_dim = image_in_dim
        self.e_dim = e_dim
        self.loss_type = loss_type
        self.quant_loss_weight = quant_loss_weight
        self.cross_modal_weight = cross_modal_weight

        self.fuse_layer_dims = [text_in_dim + image_in_dim] + layers + [e_dim]
        self.encoder = MLPLayers(layers=self.fuse_layer_dims, dropout=dropout_prob, bn=bn)

        self.rq = BalancedSinkhornRQ(
            num_emb_list, e_dim, sk_epsilons,
            kmeans_init=kmeans_init, kmeans_iters=kmeans_iters,
            sk_iters=sk_iters, use_linear=use_linear,
        )

        self.decode_layer_dims = self.fuse_layer_dims[::-1]
        self.decoder = MLPLayers(layers=self.decode_layer_dims, dropout=dropout_prob, bn=bn)

        self.text_proj = nn.Linear(text_in_dim + image_in_dim, text_in_dim)
        self.image_proj = nn.Linear(text_in_dim + image_in_dim, image_in_dim)

    def forward(self, text_x, image_x, use_sk=True):
        fused = torch.cat([text_x, image_x], dim=-1)
        x = self.encoder(fused)
        x_q, rq_loss, indices, distances = self.rq(x, use_sk=use_sk)
        out = self.decoder(x_q)

        text_recon = self.text_proj(out)
        image_recon = self.image_proj(out)

        cross_loss = F.mse_loss(text_recon, text_x) + F.mse_loss(image_recon, image_x)

        return out, rq_loss, indices, fused, text_recon, image_recon, cross_loss

    def compute_loss(self, out, quant_loss, xs=None,
                     text_x=None, image_x=None,
                     text_recon=None, image_recon=None,
                     cross_loss=None):
        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')

        loss_total = 1.2 * loss_recon + self.quant_loss_weight * quant_loss + \
                     self.cross_modal_weight * cross_loss

        return loss_total, loss_recon

    @torch.no_grad()
    def get_indices(self, text_xs, image_xs, use_sk=False):
        fused = torch.cat([text_xs, image_xs], dim=-1)
        x_e = self.encoder(fused)
        _, _, indices, distances = self.rq(x_e, use_sk=use_sk)
        return indices, distances
