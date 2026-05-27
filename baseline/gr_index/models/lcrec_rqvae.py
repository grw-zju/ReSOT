import sys
import os
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../index'))

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rqvae import RQVAE
from models.vq import VectorQuantizer
from models.rq import ResidualVectorQuantizer
from models.layers import MLPLayers, sinkhorn_algorithm


class BalancedSinkhornVQ(VectorQuantizer):

    def __init__(self, n_e, e_dim, beta=0.25, kmeans_init=False,
                 kmeans_iters=10, sk_epsilon=0.01, sk_iters=100, use_linear=0):
        super().__init__(n_e, e_dim, beta, kmeans_init, kmeans_iters,
                         sk_epsilon, sk_iters, use_linear)

    def forward(self, x, use_sk=True):
        latent = x.view(-1, self.e_dim)

        if not self.initted and self.training:
            self.init_emb(latent)

        if self.use_linear == 1:
            embeddings_weight = self.codebook_projection(self.embedding.weight)
        else:
            embeddings_weight = self.embedding.weight

        d = torch.sum(latent**2, dim=1, keepdim=True) + \
            torch.sum(embeddings_weight**2, dim=1, keepdim=True).t() - \
            2 * torch.matmul(latent, embeddings_weight.t())

        if not use_sk or self.sk_epsilon <= 0:
            indices = torch.argmin(d, dim=-1)
        else:
            d_centered = self.center_distance_for_constraint(d).double()
            Q = sinkhorn_algorithm(d_centered, self.sk_epsilon, self.sk_iters)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print("Balanced Sinkhorn returns nan/inf")
            indices = torch.argmax(Q, dim=-1)

            if self.training:
                Q = Q.to(device=embeddings_weight.device, dtype=embeddings_weight.dtype)
                xq_flat = Q @ embeddings_weight
                x_q = xq_flat.view(x.shape)

        if not (self.training and 'x_q' in locals() and use_sk and self.sk_epsilon > 0):
            if self.use_linear == 1:
                x_q = F.embedding(indices, embeddings_weight).view(x.shape)
            else:
                x_q = self.embedding(indices).view(x.shape)

        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())
        loss = codebook_loss + self.beta * commitment_loss
        x_q = x + (x_q - x).detach()
        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices, d


class BalancedSinkhornRQ(nn.Module):

    def __init__(self, n_e_list, e_dim, sk_epsilons,
                 kmeans_init=False, kmeans_iters=100, sk_iters=100, use_linear=0):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)
        self.vq_layers = nn.ModuleList([
            BalancedSinkhornVQ(n_e, e_dim, kmeans_init=kmeans_init,
                               kmeans_iters=kmeans_iters,
                               sk_epsilon=sk_epsilon, sk_iters=sk_iters,
                               use_linear=use_linear)
            for n_e, sk_epsilon in zip(n_e_list, sk_epsilons)
        ])

    def get_codebook(self):
        return torch.stack([q.get_codebook() for q in self.vq_layers])

    def forward(self, x, use_sk=True):
        all_losses = []
        all_indices = []
        all_distances = []
        x_q = 0
        residual = x
        for quantizer in self.vq_layers:
            x_res, loss, indices, distance = quantizer(residual, use_sk=use_sk)
            residual = residual - x_res
            x_q = x_q + x_res
            all_losses.append(loss)
            all_indices.append(indices)
            all_distances.append(distance)
        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)
        all_distances = torch.stack(all_distances, dim=1)
        return x_q, mean_losses, all_indices, all_distances


class LCRecRQVAE(RQVAE):

    def __init__(self, **kwargs):
        num_layers = len(kwargs.get('num_emb_list', [256, 256, 256, 256]))
        kwargs['sk_epsilons'] = [0.0] * (num_layers - 1) + [0.003]
        super().__init__(**kwargs)
        self.use_ot_loss = False
        self.ot_loss_weight = 0.0
        self.rq = BalancedSinkhornRQ(
            kwargs['num_emb_list'], kwargs['e_dim'],
            kwargs['sk_epsilons'],
            kmeans_init=kwargs.get('kmeans_init', True),
            kmeans_iters=kwargs.get('kmeans_iters', 100),
            sk_iters=kwargs.get('sk_iters', 100),
            use_linear=kwargs.get('use_linear', 0),
        )

    @torch.no_grad()
    def get_indices(self, xs, use_sk=False):
        x_e = self.encoder(xs)
        _, _, indices, distances = self.rq(x_e, use_sk=use_sk)
        return indices, distances
