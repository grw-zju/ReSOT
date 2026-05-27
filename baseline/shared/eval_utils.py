import math
import numpy as np
from collections import defaultdict


def hit_k(topk_results, k):
    hit = 0.0
    for row in topk_results:
        res = row[:k]
        if sum(res) > 0:
            hit += 1
    return hit


def ndcg_k(topk_results, k):
    ndcg = 0.0
    for row in topk_results:
        res = row[:k]
        one_ndcg = 0.0
        for i in range(len(res)):
            one_ndcg += res[i] / math.log(i + 2, 2)
        ndcg += one_ndcg
    return ndcg


def get_metrics_results(topk_results, metrics):
    res = {}
    for m in metrics:
        if m.lower().startswith("hit"):
            k = int(m.split("@")[1])
            res[m] = hit_k(topk_results, k)
        elif m.lower().startswith("ndcg"):
            k = int(m.split("@")[1])
            res[m] = ndcg_k(topk_results, k)
        else:
            raise NotImplementedError
    return res


def get_topk_results(predictions, scores, targets, k, all_items=None):
    results = []
    B = len(targets)
    predictions = [_.strip().replace(" ", "") for _ in predictions]

    if all_items is not None:
        for i, seq in enumerate(predictions):
            if seq not in all_items:
                scores[i] = -1000

    for b in range(B):
        batch_seqs = predictions[b * k: (b + 1) * k]
        batch_scores = scores[b * k: (b + 1) * k]
        pairs = [(a, b_score) for a, b_score in zip(batch_seqs, batch_scores)]
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        target_item = targets[b]
        one_results = []
        for sorted_pred in sorted_pairs:
            if sorted_pred[0] == target_item:
                one_results.append(1)
            else:
                one_results.append(0)
        results.append(one_results)

    return results


def evaluate_gr_model(predictions, labels, scores=None, all_items=None,
                      beam_size=20, k_list=[1, 5, 10]):
    if scores is not None and beam_size > 1:
        topk_results = get_topk_results(predictions, scores, labels, beam_size, all_items)
    else:
        topk_results = []
        for pred, label in zip(predictions, labels):
            pred = pred.strip().replace(" ", "")
            label = label.strip().replace(" ", "")
            if pred == label:
                topk_results.append([1])
            else:
                topk_results.append([0])

    metrics_list = [f'hit@{k}' for k in k_list] + [f'ndcg@{k}' for k in k_list]
    metrics_results = get_metrics_results(topk_results, metrics_list)

    metric = {}
    for k, v in metrics_results.items():
        metric[k.replace('@', '_at_')] = v / len(labels)

    return metric


def compute_collision_rate(indices_dict):
    all_ids = set()
    collision_count = 0
    total = len(indices_dict)

    for item_id, code in indices_dict.items():
        id_str = ''.join(code) if isinstance(code, list) else str(code)
        if id_str in all_ids:
            collision_count += 1
        all_ids.add(id_str)

    return collision_count / total if total > 0 else 0.0


def compute_knn_preservation(original_embeddings, quantized_embeddings, k=10):
    from sklearn.neighbors import NearestNeighbors

    orig_nn = NearestNeighbors(n_neighbors=k, metric='cosine').fit(original_embeddings)
    orig_neighbors = orig_nn.kneighbors(return_distance=False)

    quant_nn = NearestNeighbors(n_neighbors=k, metric='cosine').fit(quantized_embeddings)
    quant_neighbors = quant_nn.kneighbors(return_distance=False)

    jaccard_scores = []
    ndcg_scores = []

    for i in range(len(original_embeddings)):
        orig_set = set(orig_neighbors[i])
        quant_set = set(quant_neighbors[i])
        intersection = orig_set & quant_set
        union = orig_set | quant_set
        jaccard = len(intersection) / len(union) if len(union) > 0 else 0.0
        jaccard_scores.append(jaccard)

        dcg = 0.0
        for rank, neighbor in enumerate(quant_neighbors[i]):
            if neighbor in orig_set:
                orig_rank = np.where(orig_neighbors[i] == neighbor)[0][0] + 1
                rel = 1.0 / orig_rank
                dcg += rel / math.log2(rank + 2)

        idcg = sum(1.0 / (j + 1) / math.log2(j + 2) for j in range(k))
        ndcg_val = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg_val)

    return {
        f'jaccard@{k}': np.mean(jaccard_scores),
        f'ndcg@{k}': np.mean(ndcg_scores),
    }


def run_5_times(func, *args, n_runs=5, **kwargs):
    results_list = []
    for run_idx in range(n_runs):
        result = func(*args, **kwargs)
        results_list.append(result)

    mean_results = {}
    std_results = {}
    for key in results_list[0]:
        values = [r[key] for r in results_list]
        mean_results[key] = np.mean(values)
        std_results[key] = np.std(values)

    return mean_results, std_results
