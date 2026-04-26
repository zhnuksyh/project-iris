"""
src/evaluation/pairs.py

Generate genuine/impostor pairs from the test set and compute similarity scores.

Convention: all scores are **similarity** (higher = more similar).
Gabor Hamming Distances are converted to 1 - HD here.
"""

import numpy as np
from collections import defaultdict


def generate_pairs(labels: list, seed: int = 42, impostor_ratio: int = 100):
    """Generate genuine and impostor pair indices from test labels.

    Genuine pairs: all (i, j) where i < j and labels[i] == labels[j].
    Impostor pairs: random sample of different-identity pairs, sized at
    impostor_ratio * len(genuine_pairs).

    Args:
        labels: list of integer identity labels for each test sample.
        seed: random seed for reproducible impostor sampling.
        impostor_ratio: number of impostor pairs per genuine pair.

    Returns:
        (genuine_pairs, impostor_pairs) — each a list of (i, j) tuples.
    """
    # Group sample indices by identity
    identity_to_indices = defaultdict(list)
    for idx, lbl in enumerate(labels):
        identity_to_indices[lbl].append(idx)

    # Genuine pairs: all intra-class combinations
    genuine_pairs = []
    for indices in identity_to_indices.values():
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                genuine_pairs.append((indices[i], indices[j]))

    # Impostor pairs: sample inter-class pairs
    rng = np.random.RandomState(seed)
    n = len(labels)
    labels_arr = np.array(labels)
    num_impostor = len(genuine_pairs) * impostor_ratio

    impostor_pairs = []
    while len(impostor_pairs) < num_impostor:
        # Sample in batches for efficiency
        batch_size = min(num_impostor - len(impostor_pairs), num_impostor)
        idx_a = rng.randint(0, n, size=batch_size)
        idx_b = rng.randint(0, n, size=batch_size)
        # Keep only different-identity pairs
        mask = labels_arr[idx_a] != labels_arr[idx_b]
        valid = list(zip(idx_a[mask].tolist(), idx_b[mask].tolist()))
        impostor_pairs.extend(valid)

    impostor_pairs = impostor_pairs[:num_impostor]

    print(f'[pairs] Genuine pairs : {len(genuine_pairs)}')
    print(f'[pairs] Impostor pairs: {len(impostor_pairs)}')
    return genuine_pairs, impostor_pairs


def compute_cosine_scores(embeddings: np.ndarray, pairs: list) -> np.ndarray:
    """Compute cosine similarity for each pair of embeddings.

    Since embeddings are L2-normalised, cosine similarity = dot product.

    Args:
        embeddings: (N, D) array of L2-normalised embeddings.
        pairs: list of (i, j) index tuples.

    Returns:
        1-D array of similarity scores, same length as pairs.
    """
    pairs_arr = np.array(pairs)
    emb_a = embeddings[pairs_arr[:, 0]]
    emb_b = embeddings[pairs_arr[:, 1]]
    return np.sum(emb_a * emb_b, axis=1)


def compute_hamming_scores(codes: np.ndarray, pairs: list,
                           chunk_size: int = 1000) -> np.ndarray:
    """Compute similarity scores from Gabor IrisCodes using Hamming Distance.

    Returns 1 - HD so that higher values mean more similar (consistent with
    cosine similarity convention).

    Args:
        codes: (N, code_len) bool array.
        pairs: list of (i, j) index tuples.
        chunk_size: process pairs in chunks to limit memory usage.

    Returns:
        1-D array of similarity scores (1 - HD), same length as pairs.
    """
    pairs_arr = np.array(pairs)
    scores = np.empty(len(pairs_arr), dtype=np.float32)
    code_len = codes.shape[1]

    for start in range(0, len(pairs_arr), chunk_size):
        end = min(start + chunk_size, len(pairs_arr))
        chunk = pairs_arr[start:end]
        c_a = codes[chunk[:, 0]]
        c_b = codes[chunk[:, 1]]
        # Vectorized XOR + popcount
        hd = np.count_nonzero(c_a != c_b, axis=1) / code_len
        scores[start:end] = 1.0 - hd

    return scores


def compute_hamming_scores_masked(codes: np.ndarray, masks: np.ndarray,
                                  pairs: list, chunk_size: int = 1000) -> np.ndarray:
    """Hamming similarity (1 - HD) per pair, restricted to bits valid in both
    codes' masks.

    Args:
        codes: (N, code_len) bool array.
        masks: (N, code_len) bool array. True = bit is valid (use in compare).
        pairs: list of (i, j) index tuples.
        chunk_size: process pairs in chunks to limit memory usage.

    Returns:
        1-D array of similarity scores (1 - HD). For pairs whose intersection
        mask is empty, returns 0.0 (max distance == 1, similarity == 0).
    """
    pairs_arr = np.array(pairs)
    scores = np.empty(len(pairs_arr), dtype=np.float32)

    for start in range(0, len(pairs_arr), chunk_size):
        end = min(start + chunk_size, len(pairs_arr))
        chunk = pairs_arr[start:end]
        c_a = codes[chunk[:, 0]]
        c_b = codes[chunk[:, 1]]
        m_a = masks[chunk[:, 0]]
        m_b = masks[chunk[:, 1]]
        valid = m_a & m_b
        n_valid = valid.sum(axis=1)
        diffs = ((c_a ^ c_b) & valid).sum(axis=1)
        hd = np.where(n_valid > 0, diffs / np.maximum(n_valid, 1), 1.0)
        scores[start:end] = 1.0 - hd

    return scores
