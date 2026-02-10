import numpy as np

EPS = 1e-6
_EXP_LIMIT = 80  # np.log(np.finfo(np.float64).max)  # ≈ 709.78


def protected_div(left, right):
    """
    return left/right if |right| > EPS; else return left.
    """

    if np.ndim(left) == 0 and np.ndim(right) == 0:
        return left / right if abs(right) > EPS else left

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)

    out_shape = np.broadcast(left, right).shape
    out = np.empty(out_shape, dtype=np.float64)

    mask = np.abs(right) > EPS
    np.copyto(out, np.broadcast_to(left, out_shape))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        np.divide(left, right, out=out, where=mask)

    return out.item() if out.ndim == 0 else out


def protected_pow(base, exp):
    """
    ppow(b, p) = exp( clip( p * log(|b| + eps), -C, C ) )
    """
    eps = 1e-16
    C = 30

    base_is_scalar = np.isscalar(base)
    exp_is_scalar = np.isscalar(exp)

    base = np.asarray(base, dtype=float)
    exp = np.asarray(exp, dtype=float)

    abs_base = np.abs(base) + eps

    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        log_term = np.log(abs_base)
        val = exp * log_term
        clipped = np.clip(val, -C, C)
        out = np.exp(clipped)

    if base_is_scalar and exp_is_scalar:
        return float(out)
    return out


def protected_sqrt(x):
    return np.sqrt(np.abs(x))


def protected_exp(x):
    return np.exp(np.clip(x, a_min=None, a_max=_EXP_LIMIT))
