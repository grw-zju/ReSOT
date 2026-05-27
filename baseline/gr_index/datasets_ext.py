import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../index'))

import numpy as np
import torch
import torch.utils.data as data


class EmbDatasetMultimodal(data.Dataset):

    def __init__(self, data_root, datasets, text_emb_suffix='.emb-llama-td.npy',
                 image_emb_suffix='.emb-ViT-L-14.npy'):
        self.datasets = datasets.split(',')
        text_embeddings = []
        image_embeddings = []
        self.dataset_count = []

        for dataset in self.datasets:
            text_path = os.path.join(data_root, dataset, f'{dataset}{text_emb_suffix}')
            image_path = os.path.join(data_root, dataset, f'{dataset}{image_emb_suffix}')
            text_emb = np.load(text_path)
            image_emb = np.load(image_path)
            text_embeddings.append(text_emb)
            image_embeddings.append(image_emb)
            self.dataset_count.append(text_emb.shape[0])

        self.text_embeddings = np.concatenate(text_embeddings)
        self.image_embeddings = np.concatenate(image_embeddings)
        self.text_dim = self.text_embeddings.shape[-1]
        self.image_dim = self.image_embeddings.shape[-1]

    def __getitem__(self, index):
        text_emb = self.text_embeddings[index]
        image_emb = self.image_embeddings[index]
        return torch.FloatTensor(text_emb), torch.FloatTensor(image_emb)

    def __len__(self):
        return len(self.text_embeddings)
