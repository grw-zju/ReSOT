import argparse
import random
import torch
import numpy as np
from time import time
import logging
import os

from torch.utils.data import DataLoader, ConcatDataset

from datasets import EmbDataset, EmbDatasetAll
from models.rqvae import RQVAE
from trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Index")

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--eval_step', type=int, default=1, help='eval step')
    parser.add_argument('--learner', type=str, default="AdamW", help='optimizer')
    parser.add_argument("--data_root", type=str, default="", help="Input data path.")
    parser.add_argument('--datasets', type=str, default='Scientific')
    parser.add_argument('--embedding_file', type=str, default=".emb-llama-td.npy", help='')

    parser.add_argument('--weight_decay', type=float, default=1e-4, help='l2 regularization weight')
    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout ratio")
    parser.add_argument("--loss_type", type=str, default="mse", help="loss_type")
    parser.add_argument("--kmeans_init", type=int, default=1)
    parser.add_argument("--kmeans_iters", type=int, default=100)
    parser.add_argument("--bn", type=int, default=0)
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.0, 0.0],
                        help="sinkhorn epsilons")
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")

    parser.add_argument("--device", type=str, default="cuda:1", help="gpu or cpu")

    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[256, 256, 256],
                        help='emb num of every vq')
    parser.add_argument('--e_dim', type=int, default=64, help='vq codebook embedding size')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0, help='vq quantion loss weight')
    parser.add_argument('--layers', type=int, nargs='+', default=[2048, 1024, 512, 256, 128, 64],
                        help='hidden sizes of every layer')
    parser.add_argument("--ckpt_dir", type=str, default="", help="output directory for model")

    parser.add_argument('--use_ot_loss', type=int, default=1)
    parser.add_argument('--ot_loss_weight', type=float, default=0.25)
    parser.add_argument('--ot_eps', type=float, default=0.05)
    parser.add_argument('--badmm_iters', type=int, default=5)
    parser.add_argument('--badmm_kl_penalty', type=float, default=0.1)
    parser.add_argument('--fgw_alpha', type=float, default=0.25)

    parser.add_argument('--use_uot', type=int, default=1, help='use UOT soft assignment at final VQ layer')
    parser.add_argument('--uot_tau_row', type=float, default=1.0, help='UOT KL weight for item marginal (τ₁)')
    parser.add_argument('--uot_tau_col', type=float, default=0.1, help='UOT KL weight for code marginal (τ₂)')
    parser.add_argument('--uot_eta', type=float, default=0.01, help='mirror-gradient step size (η)')
    parser.add_argument('--uot_iters', type=int, default=30, help='mirror-gradient iterations')
    parser.add_argument('--early_stop_patience', type=int, default=50, help='early stop patience (in eval steps)')
    parser.add_argument('--diversity_loss_weight', type=float, default=0.1, help='diversity loss weight for codebook anti-collapse')

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

    model = RQVAE(in_dim=data.dim,
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
                  use_ot_loss=bool(args.use_ot_loss),
                  ot_loss_weight=args.ot_loss_weight,
                  ot_eps=args.ot_eps,
                  badmm_iters=args.badmm_iters,
                  badmm_kl_penalty=args.badmm_kl_penalty,
                  fgw_alpha=args.fgw_alpha,
                  use_uot=bool(args.use_uot),
                  uot_tau_row=args.uot_tau_row,
                  uot_tau_col=args.uot_tau_col,
                  uot_eta=args.uot_eta,
                  uot_iters=args.uot_iters,
                  diversity_loss_weight=args.diversity_loss_weight,
                  )
    print(model)
    data_loader = DataLoader(data, num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=True,
                             pin_memory=True)
    trainer = Trainer(args, model)
    best_loss, best_collision_rate = trainer.fit(data_loader)

    print("Best Loss", best_loss)
    print("Best Collision Rate", best_collision_rate)
