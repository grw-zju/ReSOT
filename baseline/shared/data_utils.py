import os
import json
import argparse
import numpy as np
import torch
from collections import defaultdict


def load_index_file(data_path, dataset, index_file):
    file_path = os.path.join(data_path, dataset, f'{dataset}{index_file}')
    with open(file_path, 'r') as f:
        indices = json.load(f)
    return indices


def load_inter_file(data_path, dataset):
    file_path = os.path.join(data_path, dataset, f'{dataset}.inter.json')
    with open(file_path, 'r') as f:
        inters = json.load(f)
    return inters


def load_embeddings(data_path, dataset, emb_suffix='.emb-llama-td.npy'):
    file_path = os.path.join(data_path, dataset, f'{dataset}{emb_suffix}')
    return np.load(file_path)


def load_image_embeddings(data_path, dataset, emb_suffix='.emb-ViT-L-14.npy'):
    file_path = os.path.join(data_path, dataset, f'{dataset}{emb_suffix}')
    return np.load(file_path)


def remap_inters_to_tokens(inters, indices):
    remapped = {}
    for uid, items in inters.items():
        new_items = ["".join(indices[str(i)]) for i in items]
        remapped[uid] = new_items
    return remapped


def build_seq_rec_data(remapped_inters, max_his_len=50, mode='train'):
    inter_data = []

    if mode == 'train':
        for uid in remapped_inters:
            items = remapped_inters[uid][:-2]
            for i in range(1, len(items)):
                one_data = dict()
                one_data["item"] = items[i]
                history = items[:i]
                if max_his_len > 0:
                    history = history[-max_his_len:]
                one_data["inters"] = history
                inter_data.append(one_data)

    elif mode == 'valid':
        for uid in remapped_inters:
            items = remapped_inters[uid]
            one_data = dict()
            one_data["item"] = items[-2]
            history = items[:-2]
            if max_his_len > 0:
                history = history[-max_his_len:]
            one_data["inters"] = history
            inter_data.append(one_data)

    elif mode == 'test':
        for uid in remapped_inters:
            items = remapped_inters[uid]
            one_data = dict()
            one_data["item"] = items[-1]
            history = items[:-1]
            if max_his_len > 0:
                history = history[-max_his_len:]
            one_data["inters"] = history
            inter_data.append(one_data)

    return inter_data


def get_new_tokens(indices, image_indices=None):
    new_tokens = set()
    for index in indices.values():
        for token in index:
            new_tokens.add(token)

    if image_indices is not None:
        for index in image_indices.values():
            for token in index:
                new_tokens.add(token)

    return sorted(list(new_tokens))


def get_all_items(indices):
    all_items = set()
    for index in indices.values():
        all_items.add("".join(index))
    return all_items


def generate_gr_indices(model, data_loader, device, prefix_list=None):
    if prefix_list is None:
        prefix_list = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]

    all_indices = {}
    item_idx = 0

    model.eval()
    model = model.to(device)

    with torch.no_grad():
        for data in data_loader:
            if isinstance(data, (list, tuple)):
                data = data[0]
            data = data.to(device)
            indices, _ = model.get_indices(data, use_sk=False)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()

            for index in indices:
                code = []
                for i, ind in enumerate(index):
                    code.append(prefix_list[i].format(int(ind)))
                all_indices[str(item_idx)] = code
                item_idx += 1

    return all_indices
