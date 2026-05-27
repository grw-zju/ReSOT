import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
from sklearn.cluster import KMeans


class MLPLayers(nn.Module):

    def __init__(
        self, layers, dropout=0.0, activation="relu", bn=False
    ):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(
            zip(self.layers[:-1], self.layers[1:])
        ):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            if self.use_bn:
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            activation_func = activation_layer(self.activation, output_size)
            if activation_func is not None and idx != (len(self.layers)-2):
                mlp_modules.append(activation_func)

        self.mlp_layers = nn.Sequential(*mlp_modules)
        self.apply(self.init_weights)

    def init_weights(self, module):
        # We just initialize the module with normal distribution as the paper said
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)

def activation_layer(activation_name="relu", emb_dim=None):

    if activation_name is None:
        activation = None
    elif isinstance(activation_name, str):
        if activation_name.lower() == "sigmoid":
            activation = nn.Sigmoid()
        elif activation_name.lower() == "tanh":
            activation = nn.Tanh()
        elif activation_name.lower() == "relu":
            activation = nn.ReLU()
        elif activation_name.lower() == "leakyrelu":
            activation = nn.LeakyReLU()
        elif activation_name.lower() == "none":
            activation = None
    elif issubclass(activation_name, nn.Module):
        activation = activation_name()
    else:
        raise NotImplementedError(
            "activation function {} is not implemented".format(activation_name)
        )

    return activation


@torch.no_grad()
def balanced_kmeans(samples, num_clusters, num_iters=10, sinkhorn_iters=50, epsilon=0.1):
    B, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device

    perm = torch.randperm(B, device=device)[:num_clusters]
    centers = samples[perm].clone()

    for _ in range(num_iters):
        d = torch.cdist(samples, centers, p=2) ** 2
        Q = sinkhorn_algorithm(d, epsilon, sinkhorn_iters)
        centers = (Q.T @ samples) / Q.sum(dim=0, keepdim=True).T.clamp_min(1e-12)

    return centers


@torch.no_grad()
def sinkhorn_algorithm(distances, epsilon, sinkhorn_iterations):
    Q = torch.exp(- distances / epsilon)

    B = Q.shape[0] # number of samples to assign
    K = Q.shape[1] # how many centroids per block (usually set to 256)

    # make the matrix sums to 1
    sum_Q = Q.sum(-1, keepdim=True).sum(-2, keepdim=True)
    Q /= sum_Q
    # print(Q.sum())
    for it in range(sinkhorn_iterations):

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= B

        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= K


    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q

@torch.no_grad()
def uot_mirror_gradient(cost, tau_row=1.0, tau_col=0.1, eta=0.01, iters=30):
    """
    Mirror-gradient solver for UOT (Algorithm 3, Eq.16).
    Solves: Γ = arg min_Γ≥0 <Γ,D> + τ₁ KL(Γ1∥p) + τ₂ KL(Γ^T1∥q)
    via exponentiated gradient steps with negative-entropy mirror map.

    cost: [B, K] item-to-code cost matrix
    tau_row, tau_col: KL marginal relaxation weights (τ₁, τ₂)
    eta: step size (η)
    iters: number of mirror-gradient iterations (I_uot)
    """
    B, K_num = cost.shape
    D = cost.float()

    p = torch.ones(B, device=D.device) / B
    q = torch.ones(K_num, device=D.device) / K_num

    Gamma = p.unsqueeze(1) * q.unsqueeze(0)

    for t in range(iters):
        row_sum = Gamma.sum(dim=1).clamp_min(1e-12)
        col_sum = Gamma.sum(dim=0).clamp_min(1e-12)

        log_row_ratio = torch.log(row_sum / p).clamp(max=50)
        log_col_ratio = torch.log(col_sum / q).clamp(max=50)

        grad = D + tau_row * log_row_ratio.unsqueeze(1) + tau_col * log_col_ratio.unsqueeze(0)

        Gamma = Gamma * torch.exp(-eta * grad)
        Gamma = torch.nan_to_num(Gamma, nan=0.0, posinf=0.0, neginf=0.0)

        Gamma = Gamma / Gamma.sum().clamp_min(1e-12)

    Gamma = Gamma / Gamma.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return Gamma


@torch.no_grad()
def badmm_gwot(C_s1, C_s2, C_f, a, b, alpha_fgw=0.25, beta_kl=0.1, num_iters=5):
    """
    Bregman ADMM for fused Gromov-Wasserstein OT (Algorithm 2, Eq.8-12).
    Solves: min_{T,S} <T, M(S)> + φ KL(T∥S) with T1=a, S^T1=b, T,S≥0,
    with fused cost: (1-α)M(S) + α C_f.

    C_s1: [N, N] source structure cost
    C_s2: [M, M] target structure cost
    C_f: [N, M] feature cost
    a: [N] source marginal weights
    b: [M] target marginal weights
    alpha_fgw: weight of feature cost in fused GWOT
    beta_kl: KL penalty parameter φ
    num_iters: number of BADMM iterations I_gwot

    Returns: (Gamma, S_final)
      Gamma [N, M] transport plan (symmetric average of T and S)
      S_final [N, M] final S variable for loss computation
    """
    N = C_s1.shape[0]
    M = C_s2.shape[0]
    device = C_s1.device

    T = a.unsqueeze(1) * b.unsqueeze(0)
    S = T.clone()
    U = torch.zeros(N, M, device=device, dtype=C_s1.dtype)

    for k in range(num_iters):
        M_S = C_s1 @ S @ C_s2
        G_T = (1 - alpha_fgw) * M_S + alpha_fgw * C_f + U

        T_unnorm = S * torch.exp(-G_T / beta_kl)
        row_sum = T_unnorm.sum(dim=1, keepdim=True).clamp_min(1e-12)
        T = T_unnorm / row_sum * a.unsqueeze(1)

        M_T = C_s1 @ T @ C_s2
        H_S = (1 - alpha_fgw) * M_T + alpha_fgw * C_f - U

        S_unnorm = T * torch.exp(-H_S / beta_kl)
        col_sum = S_unnorm.sum(dim=0, keepdim=True).clamp_min(1e-12)
        S = S_unnorm / col_sum * b.unsqueeze(0)

        U = U + beta_kl * (T - S)

    Gamma = (T + S) / 2
    Gamma = Gamma / Gamma.sum().clamp_min(1e-12)
    return Gamma, S
