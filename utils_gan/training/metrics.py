import sklearn.metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import torch
import numpy as np

from tqdm.auto import tqdm

def AUC(real, pred):
    return sklearn.roc_auc_score(real, pred[:,1])

def EER(real, pred, pos_label):
    try:
        fpr, tpr, threshold = sklearn.metrics.roc_curve(real,pred,pos_label=pos_label)
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr-fpr)))]
        return eer
    except:
        return None

def F1(real, pred, pos_label):
    try:
        pred_labels = (pred > 0.5).astype(float)
        return sklearn.metrics.f1_score(real, pred_labels, pos_label=pos_label)
    except:
        return None

def FAR(real, pred):
    try:
        pred_labels = (pred > 0.5).astype(float)
        fp = torch.sum((real == 0) & (pred_labels == 1))
        tn = torch.sum((real == 0) & (pred_labels == 0))
        return fp / (fp + tn)
    except:
        return None

def FRR(real, pred):
    try:
        pred_labels = (pred > 0.5).astype(float)
        fn = torch.sum((real == 1) & (pred_labels == 0))
        tp = torch.sum((real == 1) & (pred_labels == 1))
        return fn / (fn + tp)
    except:
        return None

def precision(real, pred, pos_label):
    try:
        pred_labels = (pred > 0.5).astype(float)
        return sklearn.metrics.precision_score(real, pred_labels, pos_label=pos_label)
    except:
        return None

def recall(real, pred, pos_label):
    try:
        pred_labels = (pred > 0.5).astype(float)
        return sklearn.metrics.recall_score(real, pred_labels, pos_label=pos_label)
    except:
        return None

def set_up_metrics_list(bonafide_class):
    metrics_list = [
        ['Accuracy',            lambda real, pred: np.sum((pred > 0.5).astype(int) == real.astype(int)) / len(pred)],
        # ['AUC',         lambda real, pred: sklearn.metrics.roc_auc_score(real, pred)],
        ['F1 bonafide',         lambda real, pred: F1( real, pred, pos_label=bonafide_class)],
        ['F1 spoofed',          lambda real, pred: F1( real, pred, pos_label=1-bonafide_class)],
        # ['FAR',         lambda real, pred: FAR(real, pred)],
        # ['FRR',         lambda real, pred: FRR(real, pred)],
        ['Precision bonafide',  lambda real, pred: precision(real, pred, pos_label=bonafide_class)],
        ['Precision spoofed',   lambda real, pred: precision(real, pred, pos_label=1-bonafide_class)],
        ['Recall bonafide',     lambda real, pred: recall(real, pred, pos_label=bonafide_class)],
        ['Recall spoofed',      lambda real, pred: recall(real, pred, pos_label=1-bonafide_class)],
        ['EER bonafide',                 lambda real, pred: EER(real, pred, pos_label=bonafide_class)],
        ['EER spoofed',                 lambda real, pred: EER(real, pred, pos_label=1-bonafide_class)],
    ]
    return metrics_list

def test_subset(metrics_list, full_real, full_predicted):
    metric_values = {t[0]:0 for t in metrics_list}
    for m_name, m in metrics_list:
        values = m(real=full_real, pred=full_predicted)
        metric_values[m_name] = values
    return metric_values

def test_metrics(metrics_list, predictions):
    predictor_types = list(predictions.keys())

    data_types = ["noised", "non-noised", "all"]
    metrics_values = {pt:{f'on {dt}':None for dt in data_types} for pt in predictor_types}
    
    for pt in predictions:
        for dt in data_types:
            metrics_values[pt][f"on {dt}"] = test_subset(metrics_list, predictions[pt][f"label {dt}"], predictions[pt][f"pred {dt}"])
    return metrics_values