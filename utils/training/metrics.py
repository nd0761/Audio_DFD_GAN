import sklearn.metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import torch
import numpy as np

from tqdm.auto import tqdm

def EER(real, pred):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(real, pred)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer

def F1(real, pred):
    pred_labels = (pred > 0.5).astype(float)
    return sklearn.metrics.f1_score(real, pred_labels)

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

def precision(real, pred):
    pred_labels = (pred > 0.5).astype(float)
    return sklearn.metrics.precision_score(real, pred_labels)

def recall(real, pred):
    pred_labels = (pred > 0.5).astype(float)
    return sklearn.metrics.recall_score(real, pred_labels)

metrics = [
    ['Accuracy',    lambda real, pred: np.sum((pred > 0.5).astype(int) == real.astype(int)) / len(pred)],
    # ['AUC',         lambda real, pred: sklearn.metrics.roc_auc_score(real, pred)],
    # ['EER',         lambda real, pred: EER(real, pred)],
    ['F1',          lambda real, pred: F1( real, pred)],
    # ['FAR',         lambda real, pred: FAR(real, pred)],
    # ['FRR',         lambda real, pred: FRR(real, pred)],
    ['Precision',   lambda real, pred: precision(real, pred)],
    ['Recall',      lambda real, pred: recall(real, pred)],
]

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
    print(sum(full_real), sum((full_noised > 0.5).astype(int)), sum((full_nonnoised > 0.5).astype(int)))
    print(sum(full_real), sum(full_noised), sum(full_nonnoised))
    print(len(full_real), len(full_noised), len(full_nonnoised))
    print(full_real[:10], full_noised[:10], full_nonnoised[:10])
    
    # for m_name, m in metrics:
    #     # label_num = label.cpu().detach().numpy()
    #     # pred_num_nois = np.squeeze(disc_noised_pred.cpu().detach().numpy())
    #     # pred_num_real = np.squeeze(disc_real_pred.cpu().detach().numpy())

    #     m_noised    = m(real=full_real, pred=full_noised)
    #     m_clean     = m(real=full_real, pred=full_nonnoised)
    #     m_all       = m(real=np.concatenate((full_real, full_real), axis=0), pred=np.concatenate((full_noised, full_nonnoised), axis=0))

    #     metrics_values[m_name]['on noised'].append(m_noised)
    #     metrics_values[m_name]['on clean'].append( m_clean)
    #     metrics_values[m_name]['on all'].append(   m_all)
    return test_metrics(full_real, full_noised, full_nonnoised)

def test_metrics(full_real, full_noised, full_nonnoised):
    metrics_values = {t[0]:{"on noised":[], "on clean":[], "on all":[]} for t in metrics}
    for m_name, m in metrics:
        # label_num = label.cpu().detach().numpy()
        # pred_num_nois = np.squeeze(disc_noised_pred.cpu().detach().numpy())
        # pred_num_real = np.squeeze(disc_real_pred.cpu().detach().numpy())

        m_noised    = m(real=full_real, pred=full_noised)
        m_clean     = m(real=full_real, pred=full_nonnoised)
        m_all       = m(real=np.concatenate((full_real, full_real), axis=0), pred=np.concatenate((full_noised, full_nonnoised), axis=0))

        metrics_values[m_name]['on noised'].append(m_noised)
        metrics_values[m_name]['on clean'].append( m_clean)
        metrics_values[m_name]['on all'].append(   m_all)
    return metrics_values