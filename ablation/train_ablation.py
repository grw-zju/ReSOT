import argparse
import random
import torch
import numpy as np
from time import time
import logging
import os

from torch.utils.data import DataLoader

sys_path = os.path.join(os.path.dirname(__file__), '../index')
import sys
sys.path.insert(0, sys_path)

from datasets import EmbDataset, EmbDatasetAll
from ablation_rqvae import AblationRQVAE
from trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Ablation Index Training")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--eval_step', type=int, default=1)
    parser.add_argument('--learner', type=str, default="AdamW")
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument('--datasets', type=str, default='Scientific')
    parser.add_argument('--embedding_file', type=str, default=".emb-llama-td.npy")
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument("--dropout_prob", type=float, default=0.0)
    parser.add_argument("--bn", type=int, default=0)
    parser.add_argument("--loss_type", type=str, default="mse")
    parser.add_argument("--kmeans_init", type=int, default=1)
    parser.add_argument("--kmeans_iters", type=int, default=100)
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.0, 0.0])
    parser.add_argument("--sk_iters", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[256, 256, 256, 256])
    parser.add_argument('--e_dim', type=int, default=64)
    parser.add_argument('--quant_loss_weight', type=float, default=1.0)
    parser.add_argument('--layers', type=int, nargs='+', default=[2048, 1024, 512, 256, 128, 64])
    parser.add_argument("--ckpt_dir", type=str, default="")

    parser.add_argument('--use_gwot', type=int, default=1, help='1=GWOT on, 0=off')
    parser.add_argument('--use_uot', type=int, default=1, help='1=UOT, 0=balanced Sinkhorn')
    parser.add_argument('--use_sq', type=int, default=1, help='1=soft quantization, 0=hard only')
    parser.add_argument('--ablation_name', type=str, default='v6', help='ablation variant name')

    parser.add_argument('--ot_eps', type=float, default=0.05)
    parser.add_argument('--badmm_iters', type=int, default=5)
    parser.add_argument('--badmm_kl_penalty', type=float, default=0.1)
    parser.add_argument('--fgw_alpha', type=float, default=0.25)
    parser.add_argument('--uot_tau_row', type=float, default=1.0)
    parser.add_argument('--uot_tau_col', type=float, default=0.1)
    parser.add_argument('--uot_eta', type=float, default=0.01)
    parser.add_argument('--uot_iters', type=int, default=30)
    parser.add_argument('--diversity_loss_weight', type=float, default=0.1)
    parser.add_argument('--early_stop_patience', type=int, default=50)

    return parser.parse_args()


if __name__ == '__main__':
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()
    print(args)

    logging.basicConfig(level=logging.DEBUG)

    data = EmbDatasetAll(args)

    rqvae_kwargs = dict(
        in_dim=data.dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=bool(args.bn),
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        kmeans_init=bool(args.kmeans_init),
        kmeans_iters=args.kmeans_iters,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
        ot_eps=args.ot_eps,
        badmm_iters=args.badmm_iters,
        badmm_kl_penalty=args.badmm_kl_penalty,
        fgw_alpha=args.fgw_alpha,
        uot_tau_row=args.uot_tau_row,
        uot_tau_col=args.uot_tau_col,
        uot_eta=args.uot_eta,
        uot_iters=args.uot_iters,
        diversity_loss_weight=args.diversity_loss_weight,
    )

    model = AblationRQVAE(
        use_gwot=bool(args.use_gwot),
        use_uot=bool(args.use_uot),
        use_sq=bool(args.use_sq),
        **rqvae_kwargs
    )
    print(model)
    print(f"Ablation config: GWOT={bool(args.use_gwot)}, UOT={bool(args.use_uot)}, SQ={bool(args.use_sq)}")

    data_loader = DataLoader(data, num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=True,
                             pin_memory=True)
    trainer = Trainer(args, model)
    best_loss, best_collision_rate = trainer.fit(data_loader)
    print("Best Loss", best_loss)
    print("Best Collision Rate", best_collision_rate)
