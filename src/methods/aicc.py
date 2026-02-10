def compute_aicc(aic: float, k_penalty: int, n_samples: int) -> float:
    return aic + 2 * k_penalty * (k_penalty + 1) / (n_samples - k_penalty - 1)
