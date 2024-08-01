import sklearn.metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import torch
import numpy as np

from tqdm.auto import tqdm

def AUC(real, pred):
    # fpr, tpr, thresholds = sklearn.metrics.roc_curve(real, pred)
    # eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return sklearn.roc_auc_score(real, pred[:,1])

def EER(real, pred):
    try:
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(real, pred)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    except:
        eer = None
    return eer

def F1(real, pred, pos_label=0):
    pred_labels = (pred > 0.5).astype(float)
    return sklearn.metrics.f1_score(real, pred_labels, pos_label=0)

def FAR(real, pred):
    pred_labels = (pred > 0.5).astype(float)
    fp = torch.sum((real == 0) & (pred_labels == 1))
    tn = torch.sum((real == 0) & (pred_labels == 0))
    return fp / (fp + tn)

def FRR(real, pred):
    pred_labels = (pred > 0.5).astype(float)
    fn = torch.sum((real == 1) & (pred_labels == 0))
    tp = torch.sum((real == 1) & (pred_labels == 1))
    return fn / (fn + tp)

def precision(real, pred, pos_label=0):
    pred_labels = (pred > 0.5).astype(float)
    return sklearn.metrics.precision_score(real, pred_labels, pos_label=0)

def recall(real, pred, pos_label=0):
    pred_labels = (pred > 0.5).astype(float)
    return sklearn.metrics.recall_score(real, pred_labels, pos_label=0)

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
        ['EER',                 lambda real, pred: EER(real, pred)],
    ]
    return metrics_list

def test_data(gen, disc, dataloader, device):
    gen.eval()
    disc.eval()

    # metrics_values = {t[0]:{"on noised":[], "on clean":[], "on all":[]} for t in metrics}

    pbar = tqdm(dataloader, desc='Evaluate disc')

    full_real       = None
    full_noised     = None
    full_nonnoised  = None

    for (data, sr, label, gen_type) in pbar:
        data = data.float()

        cur_batch_size = len(data)
        data           = data.to(device)
        label          = label.float().to(device)

        noised = gen(data)

        disc_noised_pred  = disc(noised)
        disc_real_pred    = disc(data)
        # print(sum(disc_noised_pred), sum(disc_real_pred))
        if full_real is None: full_real = label.cpu().detach().numpy()
        else:   full_real = np.concatenate((full_real, label.cpu().detach().numpy()))

        if full_noised is None: full_noised = np.squeeze(disc_noised_pred.cpu().detach().numpy())
        else:   full_noised = np.concatenate((full_noised, np.squeeze(disc_noised_pred.cpu().detach().numpy())))

        if full_nonnoised is None: full_nonnoised = np.squeeze(disc_real_pred.cpu().detach().numpy())
        else:   full_nonnoised = np.concatenate((full_nonnoised, np.squeeze(disc_real_pred.cpu().detach().numpy())))
            
        # break
    print('\n-----\nSUM using threshold', sum(full_real), 'noise', sum((full_noised > 0.5).astype(int)), 'clean', sum((full_nonnoised > 0.5).astype(int)))
    print('SUM as is', sum(full_real), 'noise', sum(full_noised), 'clean', sum(full_nonnoised))
    print('Lengths', len(full_real), 'noise', len(full_noised), 'clean', len(full_nonnoised))
    print('\n----\nExamples\n', full_real[:10], full_noised[:10], full_nonnoised[:10], '\n-----\n')
    return full_real, full_noised, full_nonnoised

def test_subset(metrics_list, full_real, full_predicted):
    metric_values = {t[0]:0 for t in metrics_list}
    for m_name, m in metrics_list:
        values = m(real=full_real, pred=full_predicted)
        metric_values[m_name] = values
    return metric_values

def test_metrics(metrics_list, full_real, full_noised, full_nonnoised):
    metrics_values = {"on noised": test_subset(metrics_list, full_real, full_noised),
                      "on clean":  test_subset(metrics_list, full_real, full_nonnoised),
                      "on all": test_subset(metrics_list, 
                                            np.concatenate((full_real, full_real), axis=0), 
                                            np.concatenate((full_noised, full_nonnoised), axis=0)),}
    return metrics_values