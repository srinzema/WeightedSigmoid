import numpy as np


def construct_weights(size, start_diagonal=0, start_weight=0.0, step_size=0.1):
    weights = []
    for row in range(size):
        # diagonal offset = row index
        for col in range(row, size):
            diagonal = col - row
            weight = start_weight + step_size * diagonal
            weights.append(weight)
    return np.array(weights)


def weighted_cosine(matrix_a, matrix_b, weights):
    weighted_dot = np.sum(matrix_a * matrix_b * weights)
    norm_a = np.sqrt(np.sum(weights * matrix_a**2))
    norm_b = np.sqrt(np.sum(weights * matrix_b**2))
    return weighted_dot / (norm_a * norm_b)
