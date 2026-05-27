import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../index'))

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rqvae import RQVAE
from models.layers import MLPLayers
from baseline.gr_index.models.lcrec_rqvae import BalancedSinkhornVQ, BalancedSinkhornRQ
from models.rq import ResidualVectorQuantizer


class MACRecRQVAE(nn.Module):

    def __init__(self, text_in_dim=4096, image_in_dim=768,
                 num_emb_list=None, e_dim=64, layers=None,
                 dropout_prob=0.0, bn=False, loss_type='mse',
                 quant_loss_weight=1.0, kmeans_init=True,
                 kmeans_iters=100, sk_iters=100, use_linear=0,
                 cross_align_weight=0.1):
        super().__init__()
        num_layers = len(num_emb_list) if num_emb_list else 4
        half = num_layers // 2

        self.text_in_dim = text_in_dim
        self.image_in_dim = image_in_dim
        self.e_dim = e_dim
        self.loss_type = loss_type
        self.quant_loss_weight = quant_loss_weight
        self.cross_align_weight = cross_align_weight
        self.num_emb_list = num_emb_list

        self.text_encoder = MLPLayers(
            [text_in_dim] + (layers or [2048, 1024, 512, 256, 128, 64]) + [e_dim],
            dropout=dropout_prob, bn=bn)
        self.image_encoder = MLPLayers(
            [image_in_dim] + (layers or [2048, 1024, 512, 256, 128, 64]) + [e_dim],
            dropout=dropout_prob, bn=bn)

        text_sk_eps = [0.0] * half + [0.003] * (num_layers - half)
        image_sk_eps = [0.0] * half + [0.003] * (num_layers - half)

        self.text_rq = BalancedSinkhornRQ(
            num_emb_list[:half], e_dim, text_sk_eps[:half],
            kmeans_init=kmeans_init, kmeans_iters=kmeans_iters,
            sk_iters=sk_iters, use_linear=use_linear,
        )
        self.image_rq = BalancedSinkhornRQ(
            num_emb_list[half:], e_dim, image_sk_eps[half:],
            kmeans_init=kmeans_init, kmeans_iters=kmeans_iters,
            sk_iters=sk_iters, use_linear=use_linear,
        )

        self.text_decoder = MLPLayers(
            [e_dim] + (layers or [2048, 1024, 512, 256, 128, 64])[::-1] + [text_in_dim],
            dropout=dropout_prob, bn=bn)
        self.image_decoder = MLPLayers(
            [e_dim] + (layers or [2048, 1024, 512, 256, 128, 64])[::-1] + [image_in_dim],
            dropout=dropout_prob, bn=bn)

    def forward(self, text_x, image_x, use_sk=True):
        t_enc = self.text_encoder(text_x)
        i_enc = self.image_encoder(image_x)

        t_q, t_rq_loss, t_indices, _ = self.text_rq(t_enc, use_sk=use_sk)
        i_q, i_rq_loss, i_indices, _ = self.image_rq(i_enc, use_sk=use_sk)

        t_out = self.text_decoder(t_q)
        i_out = self.image_decoder(i_q)

        total_rq_loss = (t_rq_loss + i_rq_loss) / 2
        cross_loss = F.mse_loss(t_enc, i_enc.detach()) + F.mse_loss(i_enc, t_enc.detach())

        return t_out, i_out, total_rq_loss, t_indices, i_indices, cross_loss

    def compute_loss(self, t_out, i_out, quant_loss,
                     text_x=None, image_x=None, cross_loss=None):
        if self.loss_type == 'mse':
            text_recon = F.mse_loss(t_out, text_x, reduction='mean')
            image_recon = F.mse_loss(i_out, image_x, reduction='mean')
        else:
            text_recon = F.l1_loss(t_out, text_x, reduction='mean')
            image_recon = F.l1_loss(i_out, image_x, reduction='mean')

        loss_recon = (text_recon + image_recon) / 2
        loss_total = 1.2 * loss_recon + self.quant_loss_weight * quant_loss + \
                     self.cross_align_weight * cross_loss

        return loss_total, loss_recon

    @torch.no_grad()
    def get_indices(self, text_xs, image_xs, use_sk=False):
        t_enc = self.text_encoder(text_xs)
        i_enc = self.image_encoder(image_xs)
        _, _, t_indices, t_distances = self.text_rq(t_enc, use_sk=use_sk)
        _, _, i_indices, i_distances = self.image_rq(i_enc, use_sk=use_sk)
        # concatenate text and image along layer dimension to form full num_emb_list
        indices = torch.cat([t_indices, i_indices], dim=-1)
        distances = torch.cat([t_distances, i_distances], dim=1)
        return indices, distances

    @torch.no_grad()
    def get_text_indices(self, text_xs, image_xs, use_sk=False):
        t_enc = self.text_encoder(text_xs)
        _, _, t_indices, t_distances = self.text_rq(t_enc, use_sk=use_sk)
        return t_indices, t_distances

    @torch.no_grad()
    def get_image_indices(self, text_xs, image_xs, use_sk=False):
        i_enc = self.image_encoder(image_xs)
        _, _, i_indices, i_distances = self.image_rq(i_enc, use_sk=use_sk)
        return i_indices, i_distances
