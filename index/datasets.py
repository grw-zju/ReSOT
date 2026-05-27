import numpy as np
import torch
import torch.utils.data as data
import os

class EmbDataset(data.Dataset):

    def __init__(self,data_path):

        self.data_path = data_path
        # self.embeddings = np.fromfile(data_path, dtype=np.float32).reshape(16859,-1)
        self.embeddings = np.load(data_path)
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb=torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        return len(self.embeddings)
    
class EmbDatasetAll(data.Dataset):

    def __init__(self, args):

        self.datasets = args.datasets.split(',')
        embeddings = []
        self.dataset_count = []
        for dataset in self.datasets:
            print(dataset)
            embedding_path = os.path.join(args.data_root, dataset, f'{dataset}{args.embedding_file}')
            embedding = np.load(embedding_path)
            embeddings.append(embedding)
            self.dataset_count.append(embedding.shape[0])
            
        self.embeddings = np.concatenate(embeddings)
        self.dim = self.embeddings.shape[-1]
        
        print(self.dataset_count)
        print(self.embeddings.shape[0])

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb=torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        return len(self.embeddings)
    
class EmbDatasetOne(data.Dataset):

    def __init__(self, args, dataset):


        print(dataset)
        embedding_path = os.path.join(args.data_root, dataset, f'{dataset}{args.embedding_file}')
        self.embedding = np.load(embedding_path)

        self.dim = self.embedding.shape[-1]
        
        self.data_count = self.embedding.shape[0]

        print(self.embedding.shape)

    def __getitem__(self, index):
        emb = self.embedding[index]
        tensor_emb=torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        return len(self.embedding)
