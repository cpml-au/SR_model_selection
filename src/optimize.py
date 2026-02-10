import multiprocessing as mp
import numpy as np
from scipy.optimize import minimize

from methods.mdl import negative_log_likelihood_gaussian

_worker_f_np = None
_worker_X = None
_worker_y = None
_worker_sigma = None
_worker_distribution = None


def evaluate_model(f_np, param_values, X):
    if X is None:
        return None
    X = np.array(X)
    args = [X[:, i] for i in range(X.shape[1])] + list(param_values)
    y_pred = f_np(*args)
    return np.array(y_pred, dtype=float)


def _init_worker_bfgs(f_np, X, y, sigma, distribution):
    global _worker_f_np, _worker_X, _worker_y, _worker_sigma, _worker_distribution
    _worker_f_np = f_np
    _worker_X = X
    _worker_y = y
    _worker_sigma = sigma
    _worker_distribution = distribution


def _run_single_restart_bfgs(x0):
    f_np = _worker_f_np
    X = _worker_X
    y = _worker_y
    sigma = _worker_sigma
    distribution = _worker_distribution

    if distribution != "gaussian":
        raise ValueError("Only gaussian distribution is supported.")

    def _nll_gaussian(params):
        theta = np.concatenate([np.asarray(params, dtype=float), [float(sigma) ** 2]])
        return negative_log_likelihood_gaussian(f_np, X, y, theta)

    res = minimize(
        _nll_gaussian,
        x0,
        method="BFGS",
    )
    theta_candidate = res.x
    fun_value = float(res.fun)
    return fun_value, theta_candidate


def optimise_parameters(
    f_np,
    param_syms,
    params_vals_org,
    X,
    y,
    sigma,
    distribution: str = "gaussian",
):
    n_restarts = 100
    n_jobs = None  # use cpu_count() or limit later

    if not param_syms:
        return []

    base_init = np.array(
        [
            (
                1.0
                if (v is None or (isinstance(v, float) and not np.isfinite(v)))
                else float(v)
            )
            for v in params_vals_org
        ],
        dtype=float,
    )
    dim = len(base_init)

    x0_list = []
    for r in range(n_restarts):
        if r == 0:
            x0 = base_init.copy()
        elif r < n_restarts // 2:
            x0 = base_init + np.random.normal(0.0, 1.0, size=dim)
        else:
            x0 = np.random.uniform(-10.0, 10.0, size=dim)
        x0_list.append(x0)

    if n_jobs is None:
        n_jobs = min(n_restarts, mp.cpu_count())

    global _worker_f_np, _worker_X, _worker_y, _worker_sigma, _worker_distribution

    try:
        try:
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = mp.get_context()

        with ctx.Pool(
            processes=n_jobs,
            initializer=_init_worker_bfgs,
            initargs=(f_np, X, y, sigma, distribution),
        ) as pool:
            results = pool.map(_run_single_restart_bfgs, x0_list)

        best_fun = np.inf
        best_theta = base_init.copy()

        for fun_val, theta_candidate in results:
            if fun_val < best_fun:
                best_fun = fun_val
                best_theta = theta_candidate

        return best_theta.tolist()

    finally:
        _worker_f_np = None
        _worker_X = None
        _worker_y = None
        _worker_sigma = None
        _worker_distribution = None
