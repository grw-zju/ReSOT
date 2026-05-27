import argparse
import random
import torch
import numpy as np
import logging
import os
import copy
from time import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

sys_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../index'))
import sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
script_dir = os.path.abspath(os.path.dirname(__file__))
if script_dir in sys.path:
    sys.path.remove(script_dir)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, repo_root)
sys.path.insert(0, sys_path)

from datasets import EmbDataset, EmbDatasetAll
from trainer import Trainer

from gr_index.models.tiger_rqvae import TigerRQVAE
from gr_index.models.lcrec_rqvae import LCRecRQVAE
from gr_index.models.letter_rqvae import LetterRQVAE
from gr_index.models.mmesid_rqvae import MMESIDRQVAE
from gr_index.models.mql4grec_rqvae import MQL4GRecRQVAE
from gr_index.models.macrec_rqvae import MACRecRQVAE
from gr_index.datasets_ext import EmbDatasetMultimodal


MODEL_MAP = {
    'tiger': TigerRQVAE,
    'lcrec': LCRecRQVAE,
    'letter': LetterRQVAE,
    'mmesid': MMESIDRQVAE,
    'mql4grec': MQL4GRecRQVAE,
    'macrec': MACRecRQVAE,
}


class MQL4GRecTrainer:

    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.logger = logging.getLogger()
        self.lr = args.lr
        self.device = torch.device(args.device)
        self.epochs = args.epochs
        self.eval_step = min(args.eval_step, self.epochs)
        self.ckpt_dir = args.ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.text_optimizer = torch.optim.AdamW(
            model.text_rqvae.parameters(), lr=self.lr, weight_decay=args.weight_decay)
        self.image_optimizer = torch.optim.AdamW(
            model.image_rqvae.parameters(), lr=self.lr, weight_decay=args.weight_decay)
        self.model = model.to(self.device)
        self.best_collision_rate = np.inf

    def fit(self, data_loader):
        for epoch_idx in range(self.epochs):
            self.model.text_rqvae.train()
            self.model.image_rqvae.train()
            total_loss = 0

            for data in tqdm(data_loader, desc=f'Train {epoch_idx}'):
                if isinstance(data, (list, tuple)):
                    text_data, image_data = data
                    text_data = text_data.to(self.device)
                    image_data = image_data.to(self.device)
                else:
                    text_data = data.to(self.device)
                    image_data = data.to(self.device)

                self.text_optimizer.zero_grad()
                text_out, text_rq_loss, text_indices = self.model.forward_text(text_data)
                text_loss, text_recon = self.model.compute_text_loss(text_out, text_rq_loss, xs=text_data)
                text_loss.backward()
                self.text_optimizer.step()

                self.image_optimizer.zero_grad()
                image_out, image_rq_loss, image_indices = self.model.forward_image(image_data)
                image_loss, image_recon = self.model.compute_image_loss(image_out, image_rq_loss, xs=image_data)
                image_loss.backward()
                self.image_optimizer.step()

                total_loss += (text_loss.item() + image_loss.item())

            if (epoch_idx + 1) % self.eval_step == 0:
                collision_rate = self._eval_collision(data_loader)
                print(f'Epoch {epoch_idx}, loss: {total_loss}, collision_rate: {collision_rate}')
                if collision_rate < self.best_collision_rate:
                    self.best_collision_rate = collision_rate
                    torch.save({
                        'args': self.args,
                        'epoch': epoch_idx,
                        'best_collision_rate': self.best_collision_rate,
                        'text_state_dict': self.model.text_rqvae.state_dict(),
                        'image_state_dict': self.model.image_rqvae.state_dict(),
                    }, os.path.join(self.ckpt_dir, 'best_collision_model.pth'))

        return self.best_collision_rate

    @torch.no_grad()
    def _eval_collision(self, data_loader):
        self.model.text_rqvae.eval()
        self.model.image_rqvae.eval()
        text_indices_list = []
        image_indices_list = []

        for data in data_loader:
            if isinstance(data, (list, tuple)):
                text_data, image_data = data
            else:
                text_data = image_data = data
            text_data = text_data.to(self.device)
            image_data = image_data.to(self.device)
            t_idx, _ = self.model.get_text_indices(text_data)
            i_idx, _ = self.model.get_image_indices(image_data)
            t_idx = t_idx.view(-1, t_idx.shape[-1]).cpu().numpy()
            i_idx = i_idx.view(-1, i_idx.shape[-1]).cpu().numpy()
            for idx in t_idx:
                text_indices_list.append('-'.join([str(int(x)) for x in idx]))
            for idx in i_idx:
                image_indices_list.append('-'.join([str(int(x)) for x in idx]))

        text_collision = (len(text_indices_list) - len(set(text_indices_list))) / len(text_indices_list)
        image_collision = (len(image_indices_list) - len(set(image_indices_list))) / len(image_indices_list)
        return (text_collision + image_collision) / 2


class MMESIDTrainer:

    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.logger = logging.getLogger()
        self.lr = args.lr
        self.device = torch.device(args.device)
        self.epochs = args.epochs
        self.eval_step = min(args.eval_step, self.epochs)
        self.ckpt_dir = args.ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=args.weight_decay)
        self.model = model.to(self.device)
        self.best_collision_rate = np.inf

    def fit(self, data_loader):
        for epoch_idx in range(self.epochs):
            self.model.train()
            total_loss = 0

            for data in tqdm(data_loader, desc=f'Train {epoch_idx}'):
                text_data, image_data = data
                text_data = text_data.to(self.device)
                image_data = image_data.to(self.device)

                self.optimizer.zero_grad()
                out, rq_loss, indices, fused, text_recon, image_recon, cross_loss = \
                    self.model(text_data, image_data)
                loss, loss_recon = self.model.compute_loss(
                    out, rq_loss, xs=fused,
                    text_x=text_data, image_x=image_data,
                    text_recon=text_recon, image_recon=image_recon,
                    cross_loss=cross_loss)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if (epoch_idx + 1) % self.eval_step == 0:
                collision_rate = self._eval_collision(data_loader)
                print(f'Epoch {epoch_idx}, loss: {total_loss}, collision_rate: {collision_rate}')
                if collision_rate < self.best_collision_rate:
                    self.best_collision_rate = collision_rate
                    torch.save({
                        'args': self.args,
                        'epoch': epoch_idx,
                        'best_collision_rate': self.best_collision_rate,
                        'state_dict': self.model.state_dict(),
                    }, os.path.join(self.ckpt_dir, 'best_collision_model.pth'))

        return self.best_collision_rate

    @torch.no_grad()
    def _eval_collision(self, data_loader):
        self.model.eval()
        indices_list = []
        num_sample = 0
        for data in data_loader:
            text_data, image_data = data
            text_data = text_data.to(self.device)
            image_data = image_data.to(self.device)
            num_sample += len(text_data)
            indices, _ = self.model.get_indices(text_data, image_data)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for idx in indices:
                indices_list.append('-'.join([str(int(x)) for x in idx]))
        return (num_sample - len(set(indices_list))) / num_sample


class MACRecTrainer:

    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.lr = args.lr
        self.device = torch.device(args.device)
        self.epochs = args.epochs
        self.eval_step = min(args.eval_step, self.epochs)
        self.ckpt_dir = args.ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=args.weight_decay)
        self.model = model.to(self.device)
        self.best_collision_rate = np.inf

    def fit(self, data_loader):
        for epoch_idx in range(self.epochs):
            self.model.train()
            total_loss = 0
            for data in tqdm(data_loader, desc=f'Train {epoch_idx}'):
                text_data, image_data = data
                text_data = text_data.to(self.device)
                image_data = image_data.to(self.device)
                self.optimizer.zero_grad()
                t_out, i_out, rq_loss, t_idx, i_idx, cross_loss = \
                    self.model(text_data, image_data)
                loss, loss_recon = self.model.compute_loss(
                    t_out, i_out, rq_loss,
                    text_x=text_data, image_x=image_data,
                    cross_loss=cross_loss)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if (epoch_idx + 1) % self.eval_step == 0:
                collision_rate = self._eval_collision(data_loader)
                print(f'Epoch {epoch_idx}, loss: {total_loss}, collision_rate: {collision_rate}')
                if collision_rate < self.best_collision_rate:
                    self.best_collision_rate = collision_rate
                    torch.save({
                        'args': self.args,
                        'epoch': epoch_idx,
                        'best_collision_rate': self.best_collision_rate,
                        'state_dict': self.model.state_dict(),
                    }, os.path.join(self.ckpt_dir, 'best_collision_model.pth'))

        return self.best_collision_rate

    @torch.no_grad()
    def _eval_collision(self, data_loader):
        self.model.eval()
        all_indices = []
        num_sample = 0
        for data in data_loader:
            text_data, image_data = data
            text_data = text_data.to(self.device)
            image_data = image_data.to(self.device)
            num_sample += len(text_data)
            t_idx, _ = self.model.get_text_indices(text_data, image_data)
            i_idx, _ = self.model.get_image_indices(text_data, image_data)
            t_idx = t_idx.view(-1, t_idx.shape[-1]).cpu().numpy()
            i_idx = i_idx.view(-1, i_idx.shape[-1]).cpu().numpy()
            for j in range(len(t_idx)):
                combined = list(t_idx[j]) + list(i_idx[j])
                all_indices.append('-'.join([str(int(x)) for x in combined]))
        return (num_sample - len(set(all_indices))) / num_sample


def parse_args():
    parser = argparse.ArgumentParser(description="GR Index Training")
    parser.add_argument('--baseline', type=str, default='tiger',
                        choices=['tiger', 'lcrec', 'letter', 'mmesid', 'mql4grec', 'macrec'])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--eval_step', type=int, default=50)
    parser.add_argument('--learner', type=str, default="AdamW")
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument('--datasets', type=str, default='Instruments')
    parser.add_argument('--embedding_file', type=str, default=".emb-llama-td.npy")
    parser.add_argument('--image_embedding_file', type=str, default=".emb-ViT-L-14.npy")
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument("--dropout_prob", type=float, default=0.0)
    parser.add_argument("--bn", type=bool, default=False)
    parser.add_argument("--loss_type", type=str, default="mse")
    parser.add_argument("--kmeans_init", type=bool, default=True)
    parser.add_argument("--kmeans_iters", type=int, default=100)
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.0, 0.003])
    parser.add_argument("--sk_iters", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[256, 256, 256, 256])
    parser.add_argument('--e_dim', type=int, default=64)
    parser.add_argument('--quant_loss_weight', type=float, default=1.0)
    parser.add_argument('--layers', type=int, nargs='+', default=[2048, 1024, 512, 256, 128, 64])
    parser.add_argument("--ckpt_dir", type=str, default="")
    parser.add_argument('--text_in_dim', type=int, default=4096)
    parser.add_argument('--image_in_dim', type=int, default=768)
    parser.add_argument('--orth_loss_weight', type=float, default=0.1)
    parser.add_argument('--cross_modal_weight', type=float, default=0.1)
    parser.add_argument('--cross_align_weight', type=float, default=0.1)
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

    baseline = args.baseline

    if baseline == 'mmesid':
        data = EmbDatasetMultimodal(args.data_root, args.datasets,
                                    args.embedding_file, args.image_embedding_file)
        model = MMESIDRQVAE(
            text_in_dim=data.text_dim, image_in_dim=data.image_dim,
            num_emb_list=args.num_emb_list, e_dim=args.e_dim,
            layers=args.layers, dropout_prob=args.dropout_prob,
            bn=args.bn, loss_type=args.loss_type,
            quant_loss_weight=args.quant_loss_weight,
            kmeans_init=args.kmeans_init, kmeans_iters=args.kmeans_iters,
            sk_iters=args.sk_iters, cross_modal_weight=args.cross_modal_weight,
        )
        data_loader = DataLoader(data, num_workers=args.num_workers,
                                 batch_size=args.batch_size, shuffle=True, pin_memory=True)
        trainer = MMESIDTrainer(args, model)
        trainer.fit(data_loader)

    elif baseline == 'mql4grec':
        data = EmbDatasetMultimodal(args.data_root, args.datasets,
                                    args.embedding_file, args.image_embedding_file)
        model = MQL4GRecRQVAE(
            text_in_dim=data.text_dim, image_in_dim=data.image_dim,
            num_emb_list=args.num_emb_list, e_dim=args.e_dim,
            layers=args.layers, dropout_prob=args.dropout_prob,
            bn=args.bn, loss_type=args.loss_type,
            quant_loss_weight=args.quant_loss_weight,
            kmeans_init=args.kmeans_init, kmeans_iters=args.kmeans_iters,
            sk_iters=args.sk_iters,
        )
        data_loader = DataLoader(data, num_workers=args.num_workers,
                                 batch_size=args.batch_size, shuffle=True, pin_memory=True)
        trainer = MQL4GRecTrainer(args, model)
        trainer.fit(data_loader)

    elif baseline == 'macrec':
        data = EmbDatasetMultimodal(args.data_root, args.datasets,
                                    args.embedding_file, args.image_embedding_file)
        model = MACRecRQVAE(
            text_in_dim=data.text_dim, image_in_dim=data.image_dim,
            num_emb_list=args.num_emb_list, e_dim=args.e_dim,
            layers=args.layers, dropout_prob=args.dropout_prob,
            bn=args.bn, loss_type=args.loss_type,
            quant_loss_weight=args.quant_loss_weight,
            kmeans_init=args.kmeans_init, kmeans_iters=args.kmeans_iters,
            sk_iters=args.sk_iters, cross_align_weight=args.cross_align_weight,
        )
        data_loader = DataLoader(data, num_workers=args.num_workers,
                                 batch_size=args.batch_size, shuffle=True, pin_memory=True)
        trainer = MACRecTrainer(args, model)
        trainer.fit(data_loader)

    else:
        data = EmbDatasetAll(args)
        rqvae_kwargs = dict(
            in_dim=data.dim, num_emb_list=args.num_emb_list,
            e_dim=args.e_dim, layers=args.layers,
            dropout_prob=args.dropout_prob, bn=args.bn,
            loss_type=args.loss_type,
            quant_loss_weight=args.quant_loss_weight,
            kmeans_init=args.kmeans_init, kmeans_iters=args.kmeans_iters,
            sk_epsilons=args.sk_epsilons, sk_iters=args.sk_iters,
        )
        if baseline == 'letter':
            rqvae_kwargs['orth_loss_weight'] = args.orth_loss_weight

        model_cls = MODEL_MAP[baseline]
        model = model_cls(**rqvae_kwargs)
        print(model)

        data_loader = DataLoader(data, num_workers=args.num_workers,
                                 batch_size=args.batch_size, shuffle=True, pin_memory=True)
        trainer = Trainer(args, model)
        best_loss, best_collision_rate = trainer.fit(data_loader)
        print("Best Loss", best_loss)
        print("Best Collision Rate", best_collision_rate)
