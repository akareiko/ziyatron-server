import numpy as np

def softmax(x):
    # Subtract the max for numerical stability (prevents overflow)
    x_max = np.max(x, axis=1, keepdims=True)
    x_exp = np.exp(x - x_max)
    # Compute softmax probabilities
    softmax_probs = x_exp / np.sum(x_exp, axis=1, keepdims=True)
    return softmax_probs