import numpy as np
import warnings
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

def compute_auc(ypred, ytrue):
    pred_vector = ypred.flatten()
    true_vector = ytrue.flatten()
    ap = average_precision_score(true_vector, pred_vector)
    auc = roc_auc_score(true_vector, pred_vector)
    return auc, ap