import collections
import json
import logging
import numpy as np
import torch
import copy
import argparse
import os
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader

sys_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../index'))
import sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, repo_root)
sys.path.insert(0, sys_path)

from datasets import EmbDataset, EmbDatasetAll
from models.rqvae import RQVAE

from baseline.gr_index.models.tiger_rqvae import TigerRQVAE
from baseline.gr_index.models.lcrec_rqvae import LCRecRQVAE
from baseline.gr_index.models.letter_rqvae import LetterRQVAE
from baseline.gr_index.models.mmesid_rqvae import MMESIDRQVAE
from baseline.gr_index.models.mql4grec_rqvae import MQL4GRecRQVAE
from baseline.gr_index.models.macrec_rqvae import MACRecRQVAE
from baseline.gr_index.datasets_ext import EmbDatasetMultimodal


MODEL_MAP = {
    'tiger': TigerRQVAE,
    'lcrec': LCRecRQVAE,
    'letter': LetterRQVAE,
    'mmesid': MMESIDRQVAE,
    'mql4grec': MQL4GRecRQVAE,
    'macrec': MACRecRQVAE,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate GR Baseline Indices")
    parser.add_argument('--baseline', type=str, default='tiger',
                        choices=['tiger', 'lcrec', 'letter', 'mmesid', 'mql4grec', 'macrec'])
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--content', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--data_root', type=str, default='')
    return parser.parse_args()


def check_collision(all_indices_str):
    return len(all_indices_str) == len(set(all_indices_str))


def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []
    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])
    return collision_item_groups


def resolve_collisions(all_indices, all_indices_str, all_indices_str_set,
                       all_distances, args):
    level = len(args.num_emb_list) - 1
    max_num = args.num_emb_list[0]
    sort_distances_index = np.argsort(all_distances, axis=2)

    item_min_dis = defaultdict(list)
    for item, distances in enumerate(all_distances):
        for dis in distances:
            item_min_dis[item].append(np.min(dis))

    collision_item_groups = get_collision_item(all_indices_str)

    tt = 0
    while True:
        tot_item = len(all_indices_str)
        tot_indice = len(set(all_indices_str))
        print(f'tot_item: {tot_item}, tot_indice: {tot_indice}')
        print("Collision Rate", (tot_item - tot_indice) / tot_item)

        if check_collision(all_indices_str) or tt == 2:
            break

        collision_item_groups = get_collision_item(all_indices_str)
        for collision_items in collision_item_groups:
            min_distances = []
            for item in collision_items:
                min_distances.append(item_min_dis[item][level])
            min_index = np.argsort(np.array(min_distances))

            for i, m_index in enumerate(min_index):
                if i == 0:
                    continue

                item = collision_items[m_index]
                ori_code = copy.deepcopy(all_indices[item])

                num = i
                while str(ori_code) in all_indices_str_set and num < max_num:
                    ori_code[level] = sort_distances_index[item][level][num]
                    num += 1

                for j in range(1, max_num):
                    if str(ori_code) in all_indices_str_set:
                        ori_code = copy.deepcopy(all_indices[item])
                        ori_code[level - 1] = sort_distances_index[item][level - 1][j]
                        num = 0
                        while str(ori_code) in all_indices_str_set and num < max_num:
                            ori_code[level] = sort_distances_index[item][level][num]
                            num += 1
                        if str(ori_code) not in all_indices_str_set:
                            break

                all_indices[item] = ori_code
                all_indices_str[item] = str(ori_code)
                all_indices_str_set.add(str(ori_code))

        tt += 1

    return all_indices, all_indices_str


def compute_collision_rate(indices_dict):
    all_ids = set()
    collision_count = 0
    total = len(indices_dict)
    for item_id, code in indices_dict.items():
        id_str = ''.join(code)
        if id_str in all_ids:
            collision_count += 1
        all_ids.add(id_str)
    return collision_count / total if total > 0 else 0.0


if __name__ == '__main__':
    args = parse_args()
    device = torch.device(args.device)

    if args.content == 'image':
        prefix = ["<A_{}>", "<B_{}>", "<C_{}>", "<D_{}>", "<E_{}>"]
    else:
        prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]

    ckpt = torch.load(args.ckpt_path, map_location=torch.device('cpu'), weights_only=False)
    train_args = ckpt['args']

    baseline = args.baseline

    if baseline in ['mmesid', 'macrec']:
        if baseline == 'mmesid':
            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            else:
                state_dict = ckpt['text_state_dict']
            data = EmbDatasetMultimodal(args.data_root, args.dataset,
                                        train_args.embedding_file if hasattr(train_args, 'embedding_file') else '.emb-llama-td.npy',
                                        train_args.image_embedding_file if hasattr(train_args, 'image_embedding_file') else '.emb-ViT-L-14.npy')
            model = MMESIDRQVAE(
                text_in_dim=train_args.text_in_dim if hasattr(train_args, 'text_in_dim') else 4096,
                image_in_dim=train_args.image_in_dim if hasattr(train_args, 'image_in_dim') else 768,
                num_emb_list=train_args.num_emb_list, e_dim=train_args.e_dim,
                layers=train_args.layers, dropout_prob=train_args.dropout_prob,
                bn=train_args.bn, loss_type=train_args.loss_type,
                quant_loss_weight=train_args.quant_loss_weight,
                kmeans_init=train_args.kmeans_init, kmeans_iters=train_args.kmeans_iters,
                sk_iters=train_args.sk_iters,
                cross_modal_weight=train_args.cross_modal_weight if hasattr(train_args, 'cross_modal_weight') else 0.1,
            )
        else:
            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            else:
                state_dict = ckpt.get('text_state_dict', ckpt.get('state_dict', {}))
            data = EmbDatasetMultimodal(args.data_root, args.dataset,
                                        train_args.embedding_file if hasattr(train_args, 'embedding_file') else '.emb-llama-td.npy',
                                        train_args.image_embedding_file if hasattr(train_args, 'image_embedding_file') else '.emb-ViT-L-14.npy')
            model = MACRecRQVAE(
                text_in_dim=train_args.text_in_dim if hasattr(train_args, 'text_in_dim') else 4096,
                image_in_dim=train_args.image_in_dim if hasattr(train_args, 'image_in_dim') else 768,
                num_emb_list=train_args.num_emb_list, e_dim=train_args.e_dim,
                layers=train_args.layers, dropout_prob=train_args.dropout_prob,
                bn=train_args.bn, loss_type=train_args.loss_type,
                quant_loss_weight=train_args.quant_loss_weight,
                kmeans_init=train_args.kmeans_init, kmeans_iters=train_args.kmeans_iters,
                sk_iters=train_args.sk_iters,
                cross_align_weight=train_args.cross_align_weight if hasattr(train_args, 'cross_align_weight') else 0.1,
            )
    elif baseline == 'mql4grec':
        if 'text_state_dict' in ckpt:
            state_dict = ckpt['text_state_dict']
        else:
            state_dict = ckpt['state_dict']
        data = EmbDatasetAll(train_args)
        model = MQL4GRecRQVAE(
            text_in_dim=data.dim, num_emb_list=train_args.num_emb_list,
            e_dim=train_args.e_dim, layers=train_args.layers,
            dropout_prob=train_args.dropout_prob, bn=train_args.bn,
            loss_type=train_args.loss_type,
            quant_loss_weight=train_args.quant_loss_weight,
            kmeans_init=train_args.kmeans_init, kmeans_iters=train_args.kmeans_iters,
            sk_iters=train_args.sk_iters,
        )

    else:
        state_dict = ckpt['state_dict']
        data = EmbDatasetAll(train_args)
        rqvae_kwargs = dict(
            in_dim=data.dim, num_emb_list=train_args.num_emb_list,
            e_dim=train_args.e_dim, layers=train_args.layers,
            dropout_prob=train_args.dropout_prob, bn=train_args.bn,
            loss_type=train_args.loss_type,
            quant_loss_weight=train_args.quant_loss_weight,
            kmeans_init=train_args.kmeans_init, kmeans_iters=train_args.kmeans_iters,
            sk_epsilons=train_args.sk_epsilons, sk_iters=train_args.sk_iters,
        )
        if baseline == 'letter':
            rqvae_kwargs['orth_loss_weight'] = train_args.orth_loss_weight if hasattr(train_args, 'orth_loss_weight') else 0.1
        model_cls = MODEL_MAP[baseline]
        model = model_cls(**rqvae_kwargs)

    if baseline == 'mql4grec':
        model.text_rqvae.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    data_loader = DataLoader(data, num_workers=train_args.num_workers,
                             batch_size=64, shuffle=False, pin_memory=True)

    all_indices = []
    all_indices_str = []
    all_distances = []
    all_indices_str_set = set()

    for d in tqdm(data_loader):
        if baseline in ['mmesid', 'macrec']:
            text_d, image_d = d[0].to(device), d[1].to(device)
            indices, distances = model.get_indices(text_d, image_d, use_sk=False)
        elif baseline == 'mql4grec':
            text_d = d.to(device)
            indices, distances = model.get_text_indices(text_d, use_sk=False)
        else:
            d = d.to(device)
            indices, distances = model.get_indices(d, use_sk=False)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        distances = distances.cpu().tolist()

        for index in indices:
            code = [int(ind) for ind in index]
            all_indices.append(code)
            all_indices_str.append(str(code))
            all_indices_str_set.add(str(code))

        all_distances.extend(distances)

    all_distances = np.array(all_distances)

    print(f"Total indices: {len(all_indices)}, Unique: {len(set(all_indices_str))}")
    collision_rate_before = (len(all_indices_str) - len(set(all_indices_str))) / len(all_indices_str)
    print(f"Collision rate before resolution: {collision_rate_before:.4f}")

    all_indices, all_indices_str = resolve_collisions(
        all_indices, all_indices_str, all_indices_str_set, all_distances, train_args)

    all_indices_dict = {}
    for item, indices in enumerate(all_indices):
        code = [prefix[i].format(int(ind)) for i, ind in enumerate(indices)]
        all_indices_dict[item] = code

    collision_rate_after = compute_collision_rate(all_indices_dict)
    print(f"Collision rate after resolution: {collision_rate_after:.4f}")

    output_file = os.path.join(args.output_dir, args.output_file)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_file, 'w') as fp:
        json.dump(all_indices_dict, fp, indent=4)

    print(f"Indices saved to: {output_file}")
