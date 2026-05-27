import argparse
import random
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from baseline.sr_baselines.models.sr_models import (
    SASRecModel, BERT4RecModel, FDSAModel, S3RecModel, VQRecModel, MISSRecModel
)


MODEL_MAP = {
    'sasrec': SASRecModel,
    'bert4rec': BERT4RecModel,
    'fdsa': FDSAModel,
    's3rec': S3RecModel,
    'vqrec': VQRecModel,
    'missrec': MISSRecModel,
}


class SRDataset(Dataset):

    def __init__(self, data_path, dataset, mode='train', max_seq_len=50):
        self.data_path = os.path.join(data_path, dataset)
        self.dataset = dataset
        self.mode = mode
        self.max_seq_len = max_seq_len

        with open(os.path.join(self.data_path, f'{dataset}.inter.json'), 'r') as f:
            self.inters = json.load(f)

        self.item_num = 0
        for uid, items in self.inters.items():
            for item in items:
                if item > self.item_num:
                    self.item_num = item

        if mode == 'train':
            self.data = self._build_train_data()
        elif mode == 'valid':
            self.data = self._build_valid_data()
        elif mode == 'test':
            self.data = self._build_test_data()

    def _build_train_data(self):
        data = []
        for uid, items in self.inters.items():
            for i in range(1, len(items) - 2):
                seq = items[:i]
                target = items[i]
                if len(seq) > self.max_seq_len:
                    seq = seq[-self.max_seq_len:]
                padded_seq = [0] * (self.max_seq_len - len(seq)) + seq
                data.append((padded_seq, target))
        return data

    def _build_valid_data(self):
        data = []
        for uid, items in self.inters.items():
            seq = items[:-2]
            target = items[-2]
            if len(seq) > self.max_seq_len:
                seq = seq[-self.max_seq_len:]
            padded_seq = [0] * (self.max_seq_len - len(seq)) + seq
            data.append((padded_seq, target))
        return data

    def _build_test_data(self):
        data = []
        for uid, items in self.inters.items():
            seq = items[:-1]
            target = items[-1]
            if len(seq) > self.max_seq_len:
                seq = seq[-self.max_seq_len:]
            padded_seq = [0] * (self.max_seq_len - len(seq)) + seq
            data.append((padded_seq, target))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.LongTensor(self.data[idx][0]), torch.LongTensor([self.data[idx][1]])


def compute_metrics(model, dataset, device, all_items, k_list=[1, 5, 10]):
    model.eval()
    results = defaultdict(list)

    with torch.no_grad():
        for seq, target in dataset.data:
            seq_tensor = torch.LongTensor([seq]).to(device)
            logits = model(seq_tensor)
            scores = logits[:, -1, :].squeeze()

            target_item = target
            all_scores = scores[1:dataset.item_num + 1].cpu().numpy()
            sorted_items = np.argsort(-all_scores) + 1

            where = np.where(sorted_items == target_item)[0]
            if len(where) == 0:
                continue
            rank = where[0] + 1

            for k in k_list:
                results[f'hr@{k}'].append(1.0 if rank <= k else 0.0)
                if rank <= k:
                    results[f'ndcg@{k}'].append(1.0 / np.log2(rank + 1))
                else:
                    results[f'ndcg@{k}'].append(0.0)

    metrics = {}
    for key in results:
        metrics[key] = np.mean(results[key])
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="SR Baseline Training")
    parser.add_argument('--model', type=str, default='sasrec',
                        choices=['sasrec', 'bert4rec', 'fdsa', 's3rec', 'vqrec', 'missrec'])
    parser.add_argument('--dataset', type=str, default='Instruments')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--max_seq_len', type=int, default=50)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--num_blocks', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--ckpt_dir', type=str, default='')
    parser.add_argument('--feature_dim', type=int, default=64)
    parser.add_argument('--codebook_size', type=int, default=256)
    parser.add_argument('--num_codebooks', type=int, default=4)
    parser.add_argument('--code_dim', type=int, default=64)
    parser.add_argument('--text_dim', type=int, default=4096)
    parser.add_argument('--image_dim', type=int, default=768)
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
    os.makedirs(args.ckpt_dir, exist_ok=True)

    train_data = SRDataset(args.data_path, args.dataset, mode='train', max_seq_len=args.max_seq_len)
    valid_data = SRDataset(args.data_path, args.dataset, mode='valid', max_seq_len=args.max_seq_len)
    test_data = SRDataset(args.data_path, args.dataset, mode='test', max_seq_len=args.max_seq_len)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    model_kwargs = dict(
        item_num=train_data.item_num,
        hidden_size=args.hidden_size,
        max_seq_len=args.max_seq_len,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
    )

    if args.model == 'fdsa':
        model_kwargs['feature_dim'] = args.feature_dim
    elif args.model == 'vqrec':
        model_kwargs['codebook_size'] = args.codebook_size
        model_kwargs['num_codebooks'] = args.num_codebooks
        model_kwargs['code_dim'] = args.code_dim
    elif args.model == 'missrec':
        model_kwargs['text_dim'] = args.text_dim
        model_kwargs['image_dim'] = args.image_dim

    model = MODEL_MAP[args.model](**model_kwargs)
    model = model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_hr5 = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for seq, target in tqdm(train_loader, desc=f'Epoch {epoch}'):
            seq = seq.to(args.device)
            target = target.to(args.device).squeeze(-1)

            optimizer.zero_grad()
            logits = model(seq)
            loss = criterion(logits[:, -1, :], target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            val_metrics = compute_metrics(model, valid_data, args.device, train_data.item_num)
            print(f'Epoch {epoch}, Loss: {total_loss:.4f}, Val HR@5: {val_metrics["hr@5"]:.4f}')

            if val_metrics['hr@5'] > best_hr5:
                best_hr5 = val_metrics['hr@5']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_hr5': best_hr5,
                    'args': args,
                }, os.path.join(args.ckpt_dir, 'best_model.pth'))

    ckpt = torch.load(os.path.join(args.ckpt_dir, 'best_model.pth'), weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    test_metrics = compute_metrics(model, test_data, args.device, train_data.item_num)

    print(f"=== {args.model} on {args.dataset} ===")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
