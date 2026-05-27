import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../index'))

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rqvae import RQVAE
from models.layers import MLPLayers
from baseline.gr_index.models.lcrec_rqvae import BalancedSinkhornRQ, LCRecRQVAE


class MQL4GRecRQVAE(nn.Module):

    def __init__(self, text_in_dim=4096, image_in_dim=768,
                 num_emb_list=None, e_dim=64, layers=None,
                 dropout_prob=0.0, bn=False, loss_type='mse',
                 quant_loss_weight=1.0, kmeans_init=True,
                 kmeans_iters=100, sk_iters=100, use_linear=0):
        super().__init__()
        num_layers = len(num_emb_list) if num_emb_list else 4
        sk_epsilons = [0.003] * num_layers

        self.text_rqvae = LCRecRQVAE(
            in_dim=text_in_dim, num_emb_list=num_emb_list,
            e_dim=e_dim, layers=layers, dropout_prob=dropout_prob,
            bn=bn, loss_type=loss_type, quant_loss_weight=quant_loss_weight,
            kmeans_init=kmeans_init, kmeans_iters=kmeans_iters,
            sk_epsilons=sk_epsilons, sk_iters=sk_iters,
        )

        self.image_rqvae = LCRecRQVAE(
            in_dim=image_in_dim, num_emb_list=num_emb_list,
            e_dim=e_dim, layers=layers, dropout_prob=dropout_prob,
            bn=bn, loss_type=loss_type, quant_loss_weight=quant_loss_weight,
            kmeans_init=kmeans_init, kmeans_iters=kmeans_iters,
            sk_epsilons=sk_epsilons, sk_iters=sk_iters,
        )

    def forward_text(self, x, use_sk=True):
        return self.text_rqvae(x, use_sk=use_sk)

    def forward_image(self, x, use_sk=True):
        return self.image_rqvae(x, use_sk=use_sk)

    def compute_text_loss(self, out, quant_loss, xs=None):
        return self.text_rqvae.compute_loss(out, quant_loss, xs=xs)

    def compute_image_loss(self, out, quant_loss, xs=None):
        return self.image_rqvae.compute_loss(out, quant_loss, xs=xs)

    @torch.no_grad()
    def get_text_indices(self, xs, use_sk=False):
        return self.text_rqvae.get_indices(xs, use_sk=use_sk)

    @torch.no_grad()
    def get_image_indices(self, xs, use_sk=False):
        return self.image_rqvae.get_indices(xs, use_sk=use_sk)
