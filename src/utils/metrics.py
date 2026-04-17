import numpy as np


def precision_recall_at_k(y_true, y_prob, k: int):
    """
    Compute mean Precision@k and Recall@k for multilabel predictions.

    y_true: numpy array of shape (n_samples, n_labels), multi-hot ground truth
    y_prob: numpy array of shape (n_samples, n_labels), predicted probabilities
    """
    precisions = []
    recalls = []

    for i in range(y_true.shape[0]):
        true_indices = np.where(y_true[i] > 0)[0]
        if len(true_indices) == 0:
            continue

        topk_indices = np.argsort(-y_prob[i])[:k]
        hits = len(set(true_indices).intersection(set(topk_indices)))

        precisions.append(hits / k)
        recalls.append(hits / len(true_indices))

    return float(np.mean(precisions)), float(np.mean(recalls))