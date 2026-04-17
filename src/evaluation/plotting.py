"""
src/evaluation/plotting.py

Visualization functions for Phase 6 evaluation report.

All plot functions return the matplotlib Figure object so notebooks can
display them inline.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm as scipy_norm

from src.utils.metrics import build_roc_curve, build_det_curve, compute_eer, compute_tar_at_far


# Consistent colours across all plots
COLOURS = {
    'ArcFace': '#2196F3',
    'Softmax': '#FF9800',
    'Gabor':   '#4CAF50',
}


def plot_roc_curves(results: dict, figsize=(8, 7)):
    """Plot overlaid ROC curves for all systems.

    Args:
        results: dict mapping system name → (genuine_scores, impostor_scores).
    """
    fig, ax = plt.subplots(figsize=figsize)
    for name, (gen, imp) in results.items():
        fpr, tpr = build_roc_curve(gen, imp)
        eer, _ = compute_eer(gen, imp)
        ax.plot(fpr, tpr, label=f'{name} (EER={eer:.4f})',
                color=COLOURS.get(name), linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    ax.set_xlabel('False Positive Rate (FAR)')
    ax.set_ylabel('True Positive Rate (1 - FRR)')
    ax.set_title('ROC Curves — System Comparison')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_det_curves(results: dict, figsize=(8, 7)):
    """Plot DET curves with probit-scaled axes.

    Args:
        results: dict mapping system name → (genuine_scores, impostor_scores).
    """
    fig, ax = plt.subplots(figsize=figsize)

    ticks = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    tick_labels = [f'{t*100:.1f}%' for t in ticks]
    tick_positions = scipy_norm.ppf(ticks)

    for name, (gen, imp) in results.items():
        fpr, fnr = build_det_curve(gen, imp)
        # Clip to avoid inf at 0 and 1
        fpr_c = np.clip(fpr, 1e-6, 1 - 1e-6)
        fnr_c = np.clip(fnr, 1e-6, 1 - 1e-6)
        ax.plot(scipy_norm.ppf(fpr_c), scipy_norm.ppf(fnr_c),
                label=name, color=COLOURS.get(name), linewidth=2)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel('False Positive Rate (FAR)')
    ax.set_ylabel('False Negative Rate (FRR)')
    ax.set_title('DET Curves — System Comparison')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_score_distributions(results: dict, figsize=(15, 5)):
    """Plot genuine vs impostor score histograms for each system.

    Args:
        results: dict mapping system name → (genuine_scores, impostor_scores).
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, (name, (gen, imp)) in zip(axes, results.items()):
        eer, thr = compute_eer(gen, imp)
        ax.hist(imp, bins=80, alpha=0.6, color='red', label='Impostor', density=True)
        ax.hist(gen, bins=80, alpha=0.6, color='blue', label='Genuine', density=True)
        ax.axvline(thr, color='black', linestyle='--', linewidth=1.5,
                   label=f'EER threshold={thr:.3f}')
        ax.set_title(f'{name}\nEER = {eer:.4f}')
        ax.set_xlabel('Similarity Score')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Score Distributions — Genuine vs Impostor', y=1.02, fontsize=14)
    fig.tight_layout()
    return fig


def plot_training_curves(history_paths: dict, figsize=(12, 10)):
    """Plot training loss and accuracy curves from saved history JSONs.

    Args:
        history_paths: dict mapping model name → path to JSON history file.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    for name, path in history_paths.items():
        with open(path) as f:
            hist = json.load(f)
        color = COLOURS.get(name, '#666666')
        epochs = range(1, len(hist['loss']) + 1)

        # Training loss
        axes[0, 0].plot(epochs, hist['loss'], label=name, color=color, linewidth=2)
        # Validation loss
        axes[0, 1].plot(epochs, hist['val_loss'], label=name, color=color, linewidth=2)
        # Training accuracy
        axes[1, 0].plot(epochs, hist['accuracy'], label=name, color=color, linewidth=2)
        # Validation accuracy
        axes[1, 1].plot(epochs, hist['val_accuracy'], label=name, color=color, linewidth=2)

    titles = ['Training Loss', 'Validation Loss', 'Training Accuracy', 'Validation Accuracy']
    for ax, title in zip(axes.ravel(), titles):
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('Training Curves', fontsize=14)
    fig.tight_layout()
    return fig


def plot_embedding_tsne(embeddings: np.ndarray, labels: np.ndarray,
                        title: str = 't-SNE Embeddings',
                        top_k: int = 20, figsize=(10, 8)):
    """Plot t-SNE visualization of embeddings for the top-K identities.

    Args:
        embeddings: (N, D) array.
        labels: (N,) integer label array.
        title: plot title.
        top_k: number of most-frequent identities to include.
    """
    from sklearn.manifold import TSNE

    # Select top-K identities with most samples
    unique, counts = np.unique(labels, return_counts=True)
    # Only keep identities with 2+ samples
    multi = unique[counts >= 2]
    top_ids = multi[np.argsort(-counts[np.isin(unique, multi)])][:top_k]

    mask = np.isin(labels, top_ids)
    sub_emb = embeddings[mask]
    sub_lbl = labels[mask]

    print(f'[plotting] t-SNE on {sub_emb.shape[0]} samples from {len(top_ids)} identities')

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, sub_emb.shape[0] - 1))
    coords = tsne.fit_transform(sub_emb)

    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=sub_lbl,
                         cmap='tab20', alpha=0.7, s=20)
    ax.set_title(title)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def build_comparison_table(results: dict) -> pd.DataFrame:
    """Build a summary comparison table.

    Args:
        results: dict mapping system name → (genuine_scores, impostor_scores).

    Returns:
        pandas DataFrame with EER, TAR@FAR=1%, TAR@FAR=0.1% for each system.
    """
    rows = []
    for name, (gen, imp) in results.items():
        eer, eer_thr = compute_eer(gen, imp)
        tar_1, _ = compute_tar_at_far(gen, imp, target_far=0.01)
        tar_01, _ = compute_tar_at_far(gen, imp, target_far=0.001)
        rows.append({
            'System': name,
            'EER (%)': f'{eer * 100:.2f}',
            'TAR @ FAR=1%': f'{tar_1 * 100:.2f}%',
            'TAR @ FAR=0.1%': f'{tar_01 * 100:.2f}%',
            'EER Threshold': f'{eer_thr:.4f}',
        })
    return pd.DataFrame(rows).set_index('System')
