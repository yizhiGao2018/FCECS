import logging
import numpy as np

from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, f1_score, adjusted_rand_score


def map_labels(Y_pred, Y):
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    return ind


def calculate_predict(embb_np, labels_true):
    kmeans = KMeans(n_clusters=len(np.unique(labels_true)), n_init=10)
    labels_pred = kmeans.fit_predict(embb_np)
    labels_pred = map_labels(labels_pred, labels_true)[1][labels_pred]
    return labels_pred, labels_true


def eval_for_multi_client(labels_predict, labels_true):
    acc = accuracy_score(labels_true, labels_predict)
    nmi = normalized_mutual_info_score(labels_true, labels_predict, average_method="arithmetic")
    ari = adjusted_rand_score(labels_true, labels_predict)
    f1_micro = f1_score(labels_true, labels_predict, average="micro")
    f1_macro = f1_score(labels_true, labels_predict, average="macro")
    logging.info({"acc": acc, "nmi": nmi, "ari": ari, "f1_mic": f1_micro, "f1_mac": f1_macro})
    return acc, acc
