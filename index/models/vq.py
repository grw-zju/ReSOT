import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import balanced_kmeans, sinkhorn_algorithm, uot_mirror_gradient


class VectorQuantizer(nn.Module):

    def __init__(self, n_e, e_dim,
                 beta=0.25, kmeans_init=False, kmeans_iters=10,
                 sk_epsilon=0.01, sk_iters=100, use_linear=0,
                 use_uot=False, uot_tau_row=1.0, uot_tau_col=0.1,
                 uot_eta=0.01, uot_iters=30):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters
        self.use_linear = use_linear

        self.use_uot = use_uot
        self.uot_tau_row = uot_tau_row
        self.uot_tau_col = uot_tau_col
        self.uot_eta = uot_eta
        self.uot_iters = uot_iters

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if not kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()

        if use_linear == 1:
            self.codebook_projection = torch.nn.Linear(self.e_dim, self.e_dim)
            torch.nn.init.normal_(self.codebook_projection.weight, std=self.e_dim ** -0.5)

    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)
        return z_q

    def init_emb(self, data):
        if data.shape[0] < self.n_e:
            return
        centers = balanced_kmeans(
            data,
            self.n_e,
            num_iters=self.kmeans_iters,
            sinkhorn_iters=50,
            epsilon=0.1,
        )
        self.embedding.weight.data.copy_(centers)
        self.initted = True

    @staticmethod
    def center_distance_for_constraint(distances):
        max_distance = distances.max()
        min_distance = distances.min()
        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    def forward(self, x, use_sk=True):
        latent = x.view(-1, self.e_dim)

        if not self.initted and self.training:
            self.init_emb(latent)

        if self.use_linear == 1:
            embeddings_weight = self.codebook_projection(self.embedding.weight)
        else:
            embeddings_weight = self.embedding.weight

        d = torch.sum(latent ** 2, dim=1, keepdim=True) + \
            torch.sum(embeddings_weight ** 2, dim=1, keepdim=True).t() - \
            2 * torch.matmul(latent, embeddings_weight.t())

        if self.use_uot and self.training:
            # Eq.15-18: UOT soft assignment at final layer (training only)
            # uot_mirror_gradient is @no_grad, Gamma has no gradient
            Gamma = uot_mirror_gradient(
                d,
                tau_row=self.uot_tau_row,
                tau_col=self.uot_tau_col,
                eta=self.uot_eta,
                iters=self.uot_iters,
            )
            # Eq.18: hard ID via argmax(Gamma) for VQ loss supervision
            indices = torch.argmax(Gamma, dim=-1)
            # Eq.17: soft quantization for reconstruction
            # weights are detached (from Gamma); gradients flow through embeddings_weight
            weights = Gamma / Gamma.sum(dim=1, keepdim=True).clamp_min(1e-12)
            x_q_soft = (weights @ embeddings_weight).view(x.shape)
            # straight-through: forward uses soft x_q, backward treats as identity on x
            x_q = x + (x_q_soft - x).detach()
        else:
            # standard hard argmin (inference or non-UOT layers)
            indices = torch.argmin(d, dim=-1)
            if self.use_linear == 1:
                x_q_hard = F.embedding(indices, embeddings_weight).view(x.shape)
            else:
                x_q_hard = self.embedding(indices).view(x.shape)
            # straight-through estimator
            x_q = x + (x_q_hard - x).detach()

        # VQ quantization loss: always use hard codeword
        x_q_hard_commit = self.embedding(indices).view(x.shape)
        commitment_loss = F.mse_loss(x_q_hard_commit.detach(), x)
        codebook_loss = F.mse_loss(x_q_hard_commit, x.detach())
        loss = codebook_loss + self.beta * commitment_loss

        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices, d
