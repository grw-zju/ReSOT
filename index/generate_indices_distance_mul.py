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

from datasets import EmbDataset, EmbDatasetAll, EmbDatasetOne
from models.rqvae import RQVAE

import os

def parse_args():
    parser = argparse.ArgumentParser(description="Index")
    
    parser.add_argument('--datasets', type=str, default=None)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--embedding_file', type=str, default=None)
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--content', type=str, default=None)
    parser.add_argument('--device', type=str, default="cuda:0")

    return parser.parse_args()

def partition_list(lst, partition_sizes):
    if sum(partition_sizes) != len(lst):
        return "Error: Sum of partition sizes does not match list length"
    
    partitions = []
    start = 0
    for size in partition_sizes:
        partitions.append(lst[start:start+size])
        start += size
    
    return partitions

def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str))
    return tot_item==tot_indice

def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count

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

config = parse_args()


ckpt_path = config.ckpt_path

device = torch.device(config.device)

if config.content == 'image':
    prefix = ["<A_{}>","<B_{}>","<C_{}>","<D_{}>","<E_{}>"]
else:
    prefix = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>","<e_{}>"]

ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
saved_args = ckpt["args"]
state_dict = ckpt["state_dict"]

first_dataset = config.datasets.split(',')[0]
first_data = EmbDatasetOne(config, first_dataset)
dim = first_data.dim

model = RQVAE(in_dim=dim,
                  num_emb_list=saved_args.num_emb_list,
                  e_dim=saved_args.e_dim,
                  layers=saved_args.layers,
                  dropout_prob=saved_args.dropout_prob,
                  bn=saved_args.bn,
                  loss_type=saved_args.loss_type,
                  quant_loss_weight=saved_args.quant_loss_weight,
                  kmeans_init=saved_args.kmeans_init,
                  kmeans_iters=saved_args.kmeans_iters,
                  sk_epsilons=saved_args.sk_epsilons,
                  sk_iters=saved_args.sk_iters,
                  use_ot_loss=getattr(saved_args, 'use_ot_loss', True),
                  ot_loss_weight=getattr(saved_args, 'ot_loss_weight', 0.25),
                  ot_eps=getattr(saved_args, 'ot_eps', 0.05),
                  badmm_iters=getattr(saved_args, 'badmm_iters', 5),
                  badmm_kl_penalty=getattr(saved_args, 'badmm_kl_penalty', 0.1),
                  fgw_alpha=getattr(saved_args, 'fgw_alpha', 0.25),
                  use_uot=getattr(saved_args, 'use_uot', True),
                  uot_tau_row=getattr(saved_args, 'uot_tau_row', 1.0),
                  uot_tau_col=getattr(saved_args, 'uot_tau_col', 0.1),
                  uot_eta=getattr(saved_args, 'uot_eta', 0.01),
                  uot_iters=getattr(saved_args, 'uot_iters', 30),
                  diversity_loss_weight=getattr(saved_args, 'diversity_loss_weight', 0.1),
                  )

model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
print(model)

datasets = config.datasets.split(',')
# data = EmbDatasetAll(config)

# data_loader = DataLoader(data,num_workers=args.num_workers,
#                              batch_size=64, shuffle=False,
#                              pin_memory=False)

all_indices = []
all_indices_str = []
all_distances = []
all_indices_str_set = set()

dataset_count = []

for dataset in datasets:
    data = EmbDatasetOne(config, dataset)
    
    dataset_count.append(data.data_count)
    
    data_loader = DataLoader(data, num_workers=4,
                             batch_size=64, shuffle=False,
                             pin_memory=False)

    for d in tqdm(data_loader):
        d = d.to(device)
        indices, distances = model.get_indices(d, use_sk=False)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        distances = distances.cpu().tolist()
        for index in indices:
            code = []
            for i, ind in enumerate(index):
                # code.append(prefix[i].format(int(ind)))
                code.append(int(ind))

            all_indices.append(code)
            all_indices_str.append(str(code))
            all_indices_str_set.add(str(code))
        # break
        all_distances.extend(distances)


all_distances = np.array(all_distances)

# print(all_distances)
print(all_distances.shape) ## (num, 4, 256)

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

# new_indices_set = set()

tt = 0
level = len(saved_args.num_emb_list) - 1
max_num = saved_args.num_emb_list[0]

while True:
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str))
    print("Collision Rate",(tot_item-tot_indice)/tot_item)
    
    if check_collision(all_indices_str):
        print('tt', tt)
        break

    collision_item_groups = get_collision_item(all_indices_str)
    # print(collision_item_groups)
    print('len(collision_item_groups)', len(collision_item_groups))
    
    
    for collision_items in tqdm(collision_item_groups, desc='solve collision'):
        
        min_distances = []
        for i, item in enumerate(collision_items):
            min_distances.append(item_min_dis[item][level])

        min_index = np.argsort(np.array(min_distances))
        
        for i, m_index in enumerate(min_index):
            
            if i == 0:
                continue
            
            item = collision_items[m_index]
            # print(item)
            
            ori_code = copy.deepcopy(all_indices[item])
            # print(ori_code)
            
            num = i
            while str(ori_code) in all_indices_str_set and num < max_num:

                ori_code[level] = sort_distances_index[item][level][num]
                num += 1
                # print(sort_distances_index[item][level])
                # print(ori_code)
                # print(num)
            
            ### 倒数第二层
            for i in range(1, max_num):
                if str(ori_code) in all_indices_str_set:
                    ori_code = copy.deepcopy(all_indices[item])
                    ori_code[level-1] = sort_distances_index[item][level-1][i]
                    
                num = 0
                while str(ori_code) in all_indices_str_set and num < max_num:

                    ori_code[level] = sort_distances_index[item][level][num]
                    num += 1
                    
                if str(ori_code) not in all_indices_str_set:
                    break
                
            all_indices[item] = ori_code
            all_indices_str[item] = str(ori_code)
            all_indices_str_set.add(str(ori_code))
        
        
    # if level == 2:
    #     break
    tt += 1


print("All indices number: ",len(all_indices))
print("Max number of conflicts: ", max(get_indices_count(all_indices_str).values()))

tot_item = len(all_indices_str)
tot_indice = len(set(all_indices_str))
print("Collision Rate",(tot_item-tot_indice)/tot_item)

all_indices_list = []
for item, indices in enumerate(all_indices):
    code = []
    for i, ind in enumerate(indices):
        code.append(prefix[i].format(int(ind)))
        
    all_indices_list.append(code)
    
print('check.')
all_indices_set = set([str(indices) for indices in all_indices_list])
print('check: ', len(all_indices_list) == len(all_indices_set))

assert sum(dataset_count) == len(all_indices_list)

partition_indices_list = partition_list(all_indices_list, dataset_count)

for dataset, indices_list in zip(datasets, partition_indices_list):
    
    indices_dict = {}
    for item, indices in enumerate(indices_list):
        
        indices_dict[item] = indices

    output_file = os.path.join(config.data_root, dataset, f'{dataset}{config.output_file}')

    with open(output_file, 'w') as fp:
        json.dump(indices_dict, fp, indent=4)
