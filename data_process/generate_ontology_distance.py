import argparse
import json
import numpy as np
import os
from collections import defaultdict


def build_taxonomy_tree(items):
    tree = defaultdict(set)
    all_paths = []
    for item_id, item_info in items.items():
        categories = item_info.get('categories', '')
        if not categories:
            all_paths.append([])
            continue
        path = [c.strip() for c in categories.split(',')]
        all_paths.append(path)
        for i in range(len(path)):
            prefix = tuple(path[:i + 1])
            tree[prefix].add(int(item_id))

    return all_paths, tree


def compute_lca_depth(path_a, path_b):
    depth = 0
    for i in range(min(len(path_a), len(path_b))):
        if path_a[i] == path_b[i]:
            depth += 1
        else:
            break
    return depth


def compute_ontology_distance(all_paths, max_depth=None):
    n = len(all_paths)
    if max_depth is None:
        max_depth = max(len(p) for p in all_paths) if all_paths else 1

    dist_matrix = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            lca = compute_lca_depth(all_paths[i], all_paths[j])
            max_len = max(len(all_paths[i]), len(all_paths[j]))
            if max_len == 0:
                dist = max_depth
            else:
                dist = max_depth - lca
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    return dist_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Instruments')
    parser.add_argument('--data_root', type=str, default='./data')
    args = parser.parse_args()

    item_file = os.path.join(args.data_root, args.dataset, f'{args.dataset}.item.json')
    print(f'Loading items from {item_file}')
    with open(item_file, 'r') as f:
        items = json.load(f)

    print(f'Total items: {len(items)}')

    all_paths, tree = build_taxonomy_tree(items)
    max_depth = max(len(p) for p in all_paths) if all_paths else 1
    print(f'Max taxonomy depth: {max_depth}')
    print(f'Total taxonomy nodes: {len(tree)}')

    dist_matrix = compute_ontology_distance(all_paths, max_depth)
    print(f'Distance matrix shape: {dist_matrix.shape}')
    print(f'Distance range: [{dist_matrix.min()}, {dist_matrix.max()}]')
    print(f'Mean distance: {dist_matrix.mean():.4f}')

    output_path = os.path.join(args.data_root, args.dataset, f'{args.dataset}.ontology_distance.npy')
    np.save(output_path, dist_matrix)
    print(f'Saved to {output_path}')
