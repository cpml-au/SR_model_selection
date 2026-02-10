def compute_aic(k_penalty: int, nll: float) -> float:
    return 2 * k_penalty + 2 * nll
