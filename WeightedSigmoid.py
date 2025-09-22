import numpy as np


def construct_weights(size, start_weight=0.0, step_size=0.1):
    idx = np.triu_indices(size, k=0)
    stratum_offsets = idx[1] - idx[0]
    w_base = start_weight + step_size * np.arange(size)
    return w_base[stratum_offsets]


def weighted_cosine(flat_vectors, weights):
    weighted = flat_vectors * weights
    norms = np.linalg.norm(weighted, axis=1, keepdims=True)  # shape (n_matrices,1)
    sim_matrix = weighted @ weighted.T
    sim_matrix /= norms @ norms.T
    return sim_matrix


def weighted_pearson(flat_vectors, weights):
    weighted = flat_vectors * weights
    weighted_centered = weighted - weighted.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(weighted_centered, axis=1, keepdims=True)
    sim_matrix = weighted_centered @ weighted_centered.T
    sim_matrix /= norms @ norms.T
    return sim_matrix
