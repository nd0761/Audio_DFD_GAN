import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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

# def visualize_separate(preds_noised, preds_nonnoised, save_loc):
#     # Combine predictions for the third plot
#     preds_combined = np.concatenate((preds_noised, preds_nonnoised))

#     # Define the figure with subplots
#     fig, axes = plt.subplots(1, 3, figsize=(18, 6))

#     # Plot for noised predictions
#     axes[0].hist(preds_noised, bins=30, alpha=0.7, label='Noised Predictions', density=True)
#     axes[0].set_title( 'Noised Predictions')
#     axes[0].set_xlabel('Prediction Value')
#     axes[0].set_ylabel('Density')
#     axes[0].legend()

#     # Plot for non-noised predictions
#     axes[1].hist(preds_nonnoised, bins=30, alpha=0.7, label='Non-Noised Predictions', density=True)
#     axes[1].set_title( 'Non-Noised Predictions')
#     axes[1].set_xlabel('Prediction Value')
#     axes[1].set_ylabel('Density')
#     axes[1].legend()

#     # Plot for combined predictions
#     axes[2].hist(preds_combined, bins=30, alpha=0.7, label='Combined Predictions', density=True)
#     axes[2].set_title( 'Combined Predictions')
#     axes[2].set_xlabel('Prediction Value')
#     axes[2].set_ylabel('Density')
#     axes[2].legend()

#     # Adjust layout to avoid overlap
#     plt.tight_layout()

#     # Save the figure
#     plt.savefig(save_loc)
#     plt.show()

# def visualize_separate(preds_noised, preds_nonnoised, save_loc):
#     # Combine predictions for the third plot
#     preds_combined = np.concatenate((preds_noised, preds_nonnoised))

#     # Define the figure with subplots
#     fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # 2 rows, 3 columns

#     # Determine common axis limits
#     all_preds = np.concatenate((preds_noised, preds_nonnoised, preds_combined))
#     min_pred, max_pred = all_preds.min() - 0.15, all_preds.max() + 0.15

#     # Plot for noised predictions
#     axes[0, 0].hist(preds_noised, bins=50, alpha=0.7, label='Noised Predictions', density=True)
#     axes[0, 0].set_title('Noised Predictions')
#     axes[0, 0].set_xlabel('Prediction Value')
#     axes[0, 0].set_ylabel('Density')
#     axes[0, 0].set_xlim(min_pred, max_pred)
#     axes[0, 0].legend()

#     # Plot for non-noised predictions
#     axes[0, 1].hist(preds_nonnoised, bins=50, alpha=0.7, label='Non-Noised Predictions', density=True)
#     axes[0, 1].set_title('Non-Noised Predictions')
#     axes[0, 1].set_xlabel('Prediction Value')
#     axes[0, 1].set_ylabel('Density')
#     axes[0, 1].set_xlim(min_pred, max_pred)
#     axes[0, 1].legend()

#     # Plot for combined predictions
#     axes[0, 2].hist(preds_combined, bins=50, alpha=0.7, label='Combined Predictions', density=True)
#     axes[0, 2].set_title('Combined Predictions')
#     axes[0, 2].set_xlabel('Prediction Value')
#     axes[0, 2].set_ylabel('Density')
#     axes[0, 2].set_xlim(min_pred, max_pred)
#     axes[0, 2].legend()

#     # Combined plot for first two
#     axes[1, 1].hist(preds_noised, bins=50, alpha=0.5, label='Noised', density=True, color='blue')
#     axes[1, 1].hist(preds_nonnoised, bins=50, alpha=0.5, label='Non-Noised', density=True, color='orange')
#     axes[1, 1].set_title('Combined Noised & Non-Noised')
#     axes[1, 1].set_xlabel('Prediction Value')
#     axes[1, 1].set_ylabel('Density')
#     axes[1, 1].set_xlim(min_pred, max_pred)
#     axes[1, 1].legend()

#     # Hide unused subplot
#     axes[1, 0].axis('off')
#     axes[1, 2].axis('off')

#     # Adjust layout to avoid overlap
#     plt.tight_layout()

#     # Save the figure
#     plt.savefig(save_loc)
#     plt.show()


def visualize_separate(preds_noised, preds_nonnoised, save_loc):
    # Combine predictions for the third plot
    preds_combined = np.concatenate((preds_noised, preds_nonnoised))

    # Define the figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # 2 rows, 3 columns

    # Determine common axis limits
    min_pred, max_pred = min(preds_noised.min(), preds_nonnoised.min()) - 0.2, max(preds_noised.max(), preds_nonnoised.max()) + 0.2
    min_pred, max_pred = -0.07, 1.07
    # Plot for noised predictions
    sns.histplot(preds_noised, bins=30, kde=True, ax=axes[0, 0], color='skyblue', label='Noised Predictions', stat='density')
    axes[0, 0].set_title('Noised Predictions')
    axes[0, 0].set_xlabel('Prediction Value')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_xlim(min_pred, max_pred)
    axes[0, 0].legend()

    # Plot for non-noised predictions
    sns.histplot(preds_nonnoised, bins=30, kde=True, ax=axes[0, 1], color='salmon', label='Non-Noised Predictions', stat='density')
    axes[0, 1].set_title('Non-Noised Predictions')
    axes[0, 1].set_xlabel('Prediction Value')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_xlim(min_pred, max_pred)
    axes[0, 1].legend()

    # Plot for combined predictions
    sns.histplot(preds_combined, bins=30, kde=True, ax=axes[0, 2], color='lightgreen', label='Combined Predictions', stat='density')
    axes[0, 2].set_title('Combined Predictions')
    axes[0, 2].set_xlabel('Prediction Value')
    axes[0, 2].set_ylabel('Density')
    axes[0, 2].set_xlim(min_pred, max_pred)
    axes[0, 2].legend()

    # Combined plot for first two
    sns.histplot(preds_noised,      bins=30, kde=True, ax=axes[1, 1], color='blue',   label='Noised',     stat='density', alpha=0.5)
    sns.histplot(preds_nonnoised,   bins=30, kde=True, ax=axes[1, 1], color='orange', label='Non-Noised', stat='density', alpha=0.5)
    axes[1, 1].set_title('Combined Noised & Non-Noised')
    axes[1, 1].set_xlabel('Prediction Value')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_xlim(min_pred, max_pred)
    # axes[1, 1].set_ylim(0, 1)  # Adjust the y-axis limits here
    axes[1, 1].legend()

    # Hide unused subplot
    axes[1, 0].axis('off')
    axes[1, 2].axis('off')

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Save the figure
    plt.savefig(save_loc)
    plt.show()