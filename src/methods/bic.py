import math


def compute_bic(k_penalty: int, nll: float, n_samples: int) -> float:
    return 2 * nll + k_penalty * math.log(n_samples)
