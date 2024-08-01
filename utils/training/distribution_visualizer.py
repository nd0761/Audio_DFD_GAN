import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def visualize(preds_noised, preds_nonnoised, save_loc):
    preds_combined = np.concatenate((preds_noised, preds_nonnoised))

    # preds_noised    = preds_noised.numpy()
    # preds_nonnoised = preds_nonnoised.numpy()
    # preds_combined  = preds_combined.numpy()

    plt.figure(figsize=(12, 6))

    sns.kdeplot(preds_noised,    label='Noised Predictions',     fill=True)
    sns.kdeplot(preds_nonnoised, label='Non-Noised Predictions', fill=True)
    sns.kdeplot(preds_combined,  label='Combined Predictions',   fill=True)

    # plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x * 100:.1f}%'))

    plt.title('Density Distribution of Predictions')
    plt.xlabel('Prediction Value')
    plt.ylabel('Density')
    plt.legend()

    plt.savefig(save_loc)

    plt.show()