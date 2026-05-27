import collections
import json
import logging
import numpy as np
import torch
import copy
from tqdm import tqdm
import argparse
from collections import defaultdict
from torch.utils.data import DataLoader
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../index'))

from datasets import EmbDataset, EmbDatasetAll
from ablation_rqvae import AblationRQVAE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--content', type=str, default=None, help='text or image')
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--embedding_file', type=str, default='.emb-llama-td.npy')
    parser.add_argument('--data_root', type=str, default='')
    parser.add_argument('--use_gwot', type=int, default=1)
    parser.add_argument('--use_uot', type=int, default=1)
    parser.add_argument('--use_sq', type=int, default=1)
    parser.add_argument('--ablation_name', type=str, default='v6')
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


args = parse_args()

dataset = args.dataset
ckpt_path = args.ckpt_path
output_dir = args.output_dir
output_file = args.output_file
output_file = os.path.join(output_dir, output_file)
device = torch.device(args.device)

if args.content == 'image':
    prefix = ["<A_{}>", "<B_{}>", "<C_{}>", "<D_{}>", "<E_{}>"]
else:
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]

ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
ckpt_args = ckpt["args"]
state_dict = ckpt["state_dict"]

data_path = os.path.join(args.data_root, dataset, f'{dataset}{args.embedding_file}')
data = EmbDataset(data_path)

model = AblationRQVAE(
    use_gwot=bool(args.use_gwot),
    use_uot=bool(args.use_uot),
    use_sq=bool(args.use_sq),
    in_dim=data.dim,
    num_emb_list=ckpt_args.num_emb_list,
    e_dim=ckpt_args.e_dim,
    layers=ckpt_args.layers,
    dropout_prob=ckpt_args.dropout_prob,
    bn=ckpt_args.bn,
    loss_type=ckpt_args.loss_type,
    quant_loss_weight=ckpt_args.quant_loss_weight,
    kmeans_init=ckpt_args.kmeans_init,
    kmeans_iters=ckpt_args.kmeans_iters,
    sk_epsilons=ckpt_args.sk_epsilons,
    sk_iters=ckpt_args.sk_iters,
    ot_eps=getattr(ckpt_args, 'ot_eps', 0.05),
    badmm_iters=getattr(ckpt_args, 'badmm_iters', 5),
    badmm_kl_penalty=getattr(ckpt_args, 'badmm_kl_penalty', 0.1),
    fgw_alpha=getattr(ckpt_args, 'fgw_alpha', 0.25),
    uot_tau_row=getattr(ckpt_args, 'uot_tau_row', 1.0),
    uot_tau_col=getattr(ckpt_args, 'uot_tau_col', 0.1),
    uot_eta=getattr(ckpt_args, 'uot_eta', 0.01),
    uot_iters=getattr(ckpt_args, 'uot_iters', 30),
)

model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

data_loader = DataLoader(data, num_workers=ckpt_args.num_workers,
                         batch_size=64, shuffle=False, pin_memory=True)

all_indices = []
all_indices_str = []
all_distances = []
all_indices_str_set = set()

for d in tqdm(data_loader):
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
sort_distances_index = np.argsort(all_distances, axis=2)

item_min_dis = defaultdict(list)
for item, distances in tqdm(enumerate(all_distances), desc='cal distances'):
    for dis in distances:
        item_min_dis[item].append(np.min(dis))

collision_item_groups = get_collision_item(all_indices_str)
all_collision_items = set()
for collision_items in collision_item_groups:
    for item in collision_items:
        all_collision_items.add(item)

print('collision items num: ', len(all_collision_items))

tt = 0
level = len(ckpt_args.num_emb_list) - 1
max_num = ckpt_args.num_emb_list[0]

while True:
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str))
    print(f'tot_item: {tot_item}, tot_indice: {tot_indice}')
    print("Collision Rate", (tot_item - tot_indice) / tot_item)

    if check_collision(all_indices_str) or tt == 2:
        print('tt', tt)
        break

    collision_item_groups = get_collision_item(all_indices_str)

    for collision_items in collision_item_groups:
        min_distances = []
        for i, item in enumerate(collision_items):
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

            for i2 in range(1, max_num):
                if str(ori_code) in all_indices_str_set:
                    ori_code = copy.deepcopy(all_indices[item])
                    ori_code[level - 1] = sort_distances_index[item][level - 1][i2]
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

print("All indices number: ", len(all_indices))
all_indices_str = [str(indice) for indice in all_indices]
print("Max number of conflicts: ", max(collections.defaultdict(int, {c: all_indices_str.count(c) for c in set(all_indices_str)}).values()))

tot_item = len(all_indices_str)
tot_indice = len(set(all_indices_str))
print("Collision Rate", (tot_item - tot_indice) / tot_item)

all_indices_dict = {}
for item, indices in enumerate(all_indices):
    code = []
    for i, ind in enumerate(indices):
        code.append(prefix[i].format(int(ind)))
    all_indices_dict[item] = code

code2item = {}
for item, code in all_indices_dict.items():
    code2item[str(code)] = item

print('check: ', len(code2item) == len(all_indices_dict))

with open(output_file, 'w') as fp:
    json.dump(all_indices_dict, fp, indent=4)
