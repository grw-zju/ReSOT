import json
import argparse
import os
import random
import numpy as np


def generate_random_ids(item_num, token_length=4, vocab_size=256, prefix_list=None):
    if prefix_list is None:
        prefix_list = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>"]

    indices = {}
    assigned_ids = set()

    for item_id in range(item_num):
        while True:
            code = []
            for i in range(token_length):
                code.append(prefix_list[i].format(random.randint(0, vocab_size - 1)))
            id_str = ''.join(code)
            if id_str not in assigned_ids:
                assigned_ids.add(id_str)
                break

        indices[str(item_id)] = code

    return indices


def generate_sequential_ids(item_num, token_length=4, vocab_size=256, prefix_list=None):
    if prefix_list is None:
        prefix_list = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>"]

    indices = {}
    for item_id in range(item_num):
        code = []
        remaining = item_id
        for i in range(token_length - 1):
            digit = remaining % vocab_size
            code.append(prefix_list[i].format(digit))
            remaining = remaining // vocab_size
        code.append(prefix_list[token_length - 1].format(remaining))

        indices[str(item_id)] = code

    return indices


def generate_cid_ids(item_num, item_titles=None, token_length=4, vocab_size=256, prefix_list=None):
    if prefix_list is None:
        prefix_list = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>"]

    indices = {}
    for item_id in range(item_num):
        code = []
        for i in range(token_length):
            if item_titles is not None and str(item_id) in item_titles:
                hash_val = hash(item_titles[str(item_id)]) % vocab_size
                code.append(prefix_list[i].format((hash_val + i) % vocab_size))
            else:
                code.append(prefix_list[i].format(item_id % vocab_size))

        indices[str(item_id)] = code

    return indices


def main():
    parser = argparse.ArgumentParser(description="P5-CID/VIP5 ID Generation")
    parser.add_argument('--dataset', type=str, default='Instruments')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--method', type=str, default='p5cid',
                        choices=['p5cid', 'vip5'])
    parser.add_argument('--token_length', type=int, default=4)
    parser.add_argument('--vocab_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


if __name__ == '__main__':
    args = main()
    random.seed(args.seed)
    np.random.seed(args.seed)

    data_path = os.path.join(args.data_path, args.dataset)
    output_dir = args.output_dir or data_path
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(data_path, f'{args.dataset}.inter.json'), 'r') as f:
        inters = json.load(f)

    item_num = 0
    all_items = set()
    for uid, items in inters.items():
        for item in items:
            all_items.add(item)
            if item > item_num:
                item_num = item

    if args.method == 'p5cid':
        indices = generate_cid_ids(item_num + 1, token_length=args.token_length,
                                   vocab_size=args.vocab_size)
    elif args.method == 'vip5':
        indices = generate_random_ids(item_num + 1, token_length=args.token_length,
                                     vocab_size=args.vocab_size)

    output_file = os.path.join(output_dir, f'{args.dataset}.{args.method}_indices.json')
    with open(output_file, 'w') as f:
        json.dump(indices, f, indent=4)

    collision_count = 0
    id_set = set()
    for item_id, code in indices.items():
        id_str = ''.join(code)
        if id_str in id_set:
            collision_count += 1
        id_set.add(id_str)

    collision_rate = collision_count / len(indices)
    print(f'{args.method} ID generation completed for {args.dataset}')
    print(f'Total items: {len(indices)}, Collision rate: {collision_rate:.4f}')
    print(f'Output saved to: {output_file}')
