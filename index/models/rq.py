import torch
import torch.nn as nn

from .vq import VectorQuantizer


class ResidualVectorQuantizer(nn.Module):

    def __init__(self, n_e_list, e_dim, sk_epsilons,
                 kmeans_init=False, kmeans_iters=100, sk_iters=100, use_linear=0,
                 use_uot=False, uot_tau_row=1.0, uot_tau_col=0.1,
                 uot_eta=0.01, uot_iters=30):
        super().__init__()
        assert len(n_e_list) == len(sk_epsilons), \
            f"num_emb_list length ({len(n_e_list)}) must match sk_epsilons length ({len(sk_epsilons)})"
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters

        self.vq_layers = nn.ModuleList([
            VectorQuantizer(
                n_e, e_dim,
                kmeans_init=self.kmeans_init,
                kmeans_iters=self.kmeans_iters,
                sk_epsilon=sk_epsilon,
                sk_iters=sk_iters,
                use_linear=use_linear,
                use_uot=(use_uot and i == len(n_e_list) - 1),
                uot_tau_row=uot_tau_row,
                uot_tau_col=uot_tau_col,
                uot_eta=uot_eta,
                uot_iters=uot_iters,
            )
            for i, (n_e, sk_epsilon) in enumerate(zip(n_e_list, sk_epsilons))
        ])

    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            all_codebook.append(quantizer.get_codebook())
        return torch.stack(all_codebook)

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
