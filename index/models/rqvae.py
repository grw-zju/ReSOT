import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .layers import MLPLayers, badmm_gwot
from .rq import ResidualVectorQuantizer


class RQVAE(nn.Module):
    def __init__(self,
                 in_dim=768,
                 num_emb_list=None,
                 e_dim=64,
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 kmeans_init=False,
                 kmeans_iters=100,
                 sk_epsilons=None,
                 sk_iters=100,
                 use_linear=0,
                 use_ot_loss=True,
                 ot_loss_weight=0.25,
                 ot_eps=0.05,
                 badmm_iters=5,
                 badmm_kl_penalty=0.1,
                 fgw_alpha=0.25,
                 use_uot=True,
                 uot_tau_row=1.0,
                 uot_tau_col=0.1,
                 uot_eta=0.01,
                 uot_iters=30,
                 diversity_loss_weight=0.1,
                 ):
        super(RQVAE, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim

        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight = quant_loss_weight
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters

        self.use_ot_loss = use_ot_loss
        self.ot_loss_weight = ot_loss_weight
        self.ot_eps = ot_eps
        self.badmm_iters = badmm_iters
        self.badmm_kl_penalty = badmm_kl_penalty
        self.fgw_alpha = fgw_alpha

        self.use_uot = use_uot
        self.uot_tau_row = uot_tau_row
        self.uot_tau_col = uot_tau_col
        self.uot_eta = uot_eta
        self.uot_iters = uot_iters

        self.diversity_loss_weight = diversity_loss_weight

        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob, bn=self.bn)

        self.rq = ResidualVectorQuantizer(num_emb_list, e_dim,
                                          kmeans_init=self.kmeans_init,
                                          kmeans_iters=self.kmeans_iters,
                                          sk_epsilons=self.sk_epsilons,
                                          sk_iters=self.sk_iters,
                                          use_linear=use_linear,
                                          use_uot=self.use_uot,
                                          uot_tau_row=self.uot_tau_row,
                                          uot_tau_col=self.uot_tau_col,
                                          uot_eta=self.uot_eta,
                                          uot_iters=self.uot_iters)

        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                                 dropout=self.dropout_prob, bn=self.bn)

    def forward(self, x, use_sk=True):
        x_enc = self.encoder(x)
        x_q, rq_loss, indices, distances = self.rq(x_enc, use_sk=use_sk)
        out = self.decoder(x_q)
        return out, rq_loss, indices

    @torch.no_grad()
    def get_indices(self, xs, use_sk=False):
        x_e = self.encoder(xs)
        _, _, indices, distances = self.rq(x_e, use_sk=use_sk)
        return indices, distances

    def codebook_diversity_loss(self):
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for quantizer in self.rq.vq_layers:
            cb = quantizer.embedding.weight
            pairwise_dist = torch.cdist(cb, cb, p=2)
            mean_dist = pairwise_dist.mean()
            total_loss = total_loss - mean_dist
        return total_loss / len(self.rq.vq_layers)

    @torch.no_grad()
    def revive_dead_codes(self, x_enc):
        residual = x_enc
        for quantizer in self.rq.vq_layers:
            emb_w = quantizer.embedding.weight
            d = torch.sum(residual**2, dim=1, keepdim=True) + \
                torch.sum(emb_w**2, dim=1, keepdim=True).t() - \
                2 * torch.matmul(residual, emb_w.t())
            indices = torch.argmin(d, dim=-1)
            active = torch.bincount(indices, minlength=quantizer.n_e) > 0
            dead_indices = (~active).nonzero(as_tuple=True)[0]
            n_dead = dead_indices.shape[0]
            if n_dead > 0:
                n_replace = min(n_dead, residual.shape[0])
                random_idx = torch.randperm(residual.shape[0], device=residual.device)[:n_replace]
                random_residuals = residual[random_idx].detach()
                quantizer.embedding.weight.data[dead_indices[:n_replace]] = random_residuals
            x_res = quantizer.embedding(indices)
            residual = residual - x_res

    def compute_loss(self, out, quant_loss, xs=None):
        if self.loss_type == 'mse':
            loss_recon_point = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon_point = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')

        if self.use_ot_loss:
            loss_recon_ot = self._fused_gwot_loss(out, xs)
        else:
            loss_recon_ot = out.new_tensor(0.0)

        loss_recon = loss_recon_point + self.ot_loss_weight * loss_recon_ot

        loss_total = loss_recon + self.quant_loss_weight * quant_loss

        if self.diversity_loss_weight > 0:
            div_loss = self.codebook_diversity_loss()
            loss_total = loss_total + self.diversity_loss_weight * div_loss

        return loss_total, loss_recon

    def _fused_gwot_loss(self, out, xs):
        N = xs.shape[0]
        xs_detach = xs.detach()

        C_s1 = torch.cdist(xs_detach, xs_detach, p=2) ** 2
        C_s1 = self.ot_eps * C_s1

        C_s2 = torch.cdist(out, out, p=2) ** 2
        C_s2 = self.ot_eps * C_s2

        C_f = torch.cdist(xs_detach, out, p=2) ** 2
        C_f = self.ot_eps * C_f

        a = torch.ones(N, device=xs.device, dtype=torch.float32) / N
        b = torch.ones(N, device=xs.device, dtype=torch.float32) / N

        Gamma, S_final = badmm_gwot(C_s1, C_s2, C_f, a, b,
                           alpha_fgw=self.fgw_alpha,
                           beta_kl=self.badmm_kl_penalty,
                           num_iters=self.badmm_iters)

        M_S = C_s1 @ S_final @ C_s2
        gw_loss = torch.sum(Gamma * ((1 - self.fgw_alpha) * M_S + self.fgw_alpha * C_f))

        return gw_loss

