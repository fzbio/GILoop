import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_val_score(adj_pred, adj_orig, edges_pos, edges_neg):
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_pred[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_pred[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    acc = get_accuracy(preds_all, labels_all, from_logits=False)

    return roc_score, ap_score, acc


def get_accuracy(preds, labels, from_logits=True):
    if from_logits:
        preds = sigmoid(preds)
    labels = labels.astype(int)
    return np.mean(np.equal(labels, np.round(preds)))