import sys
import math
import numpy as np
import sympy as sp
from pathlib import Path
import pandas as pd
from scipy.optimize import minimize, least_squares
from sklearn.ensemble import RandomForestRegressor
import multiprocessing as mp
from expressions import (
    CUSTOM_NUMPY,
    load_operon,
    count_number_of_nodes,
    log_functional_spec,
)

np.random.seed(42)

# ############# Loading models and data-apoints

# Change the value of sigma in theta_full and nll function based on the true sigma
n_points = 10100
n_level = 0.1
models_path = {
    "f1": "f1_100_10.operon",
    "f2": "f2_100_2.operon",
    "f3": "f3_100_1.operon",
    "f4": "f4_100_2.operon",
    "f5": "f5_100_3.operon",
    "f6": "f6_100_2.operon",
    "f7": "f7_100_2.operon",
}


data_points_path = {
    "f1": f"friedman_{n_points}_noise-{n_level}",
    "f2": f"kotanchek_{n_points}_noise-{n_level}",
    "f3": f"salustowicz_{n_points}_noise-{n_level}",
    "f4": f"salustowicz2d_{n_points}_noise-{n_level}",
    "f5": f"ratpol3d_{n_points}_noise-{n_level}",
    "f6": f"ratpol2d_{n_points}_noise-{n_level}",
    "f7": f"ripple_{n_points}_noise-{n_level}",
}


# ---------------------- CLI selection of functions ----------------------
cli_funcs = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

if cli_funcs:
    unknown = [f for f in cli_funcs if f not in models_path]
    if unknown:
        raise SystemExit(
            f"Unknown function(s): {unknown}. "
            f"Valid choices are: {list(models_path.keys())}"
        )
    models_path = {k: v for k, v in models_path.items() if k in cli_funcs}
    data_points_path = {k: v for k, v in data_points_path.items() if k in cli_funcs}

# ##################### Load Models


def evaluate_model(f_np, var_symbols, param_values, X):
    if X is None:
        return None
    X = np.array(X)
    args = [X[:, i] for i in range(X.shape[1])] + list(param_values)
    y_pred = f_np(*args)
    return np.array(y_pred, dtype=float)


def compute_SSE(y_true, y_pred):
    if y_true is None or y_pred is None:
        return None
    err = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)

    clip = 1e150  # clip the value to prevent overflow
    err = np.nan_to_num(err, nan=clip, posinf=clip, neginf=-clip)
    err = np.clip(err, -clip, clip)  # prevent overflow in square
    return float(np.dot(err, err))


def compute_MSE(y_true, y_pred):
    return compute_SSE(y_true, y_pred) / len(y_true)


def reorder_X(X, var_symbols, all_columns):
    """Return a view of X whose columns match the order of var_symbols."""
    idx = [all_columns.index(sym.name) for sym in var_symbols]
    return X[:, idx]


def negative_log_likelihood(param_values, f_np, X, y, distribution, sigma):
    # sigma = np.sqrt(0.1)
    y_pred = evaluate_model(f_np, None, param_values, X)

    # bail out early if predictions blow up
    if (not np.all(np.isfinite(y_pred))) or np.any(np.abs(y_pred) > 1e150):
        return 1e40

    y_true = np.asarray(y, dtype=float)
    N = len(y_true)

    if distribution == "gaussian":
        # sse = np.sum((y_true - y_pred) ** 2)
        sse = compute_SSE(y_true=y_true, y_pred=y_pred)
        sse = max(sse, 1e-12)  # avoid log(0)
        if sigma is None:
            return 0.5 * N * (np.log(2 * np.pi) + 1.0 + np.log(sse / N))
        else:
            # fixed sigma
            sigma2 = float(sigma) ** 2
            return 0.5 * N * np.log(2 * np.pi * sigma2) + 0.5 * sse / sigma2

    elif distribution == "bernoulli":
        p = 1.0 / (1.0 + np.exp(-y_pred))
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return -np.sum(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    elif distribution == "poisson":
        lam = np.exp(
            np.clip(y_pred, -50, 50)
        )  # y_pre is the input. keep λ in [e-50, e50]
        return np.sum(lam - y_true * np.log(lam.clip(1e-12)))

    raise ValueError("unknown distribution")


def _fit_params_lm(f_np, param_syms, params_vals_org, X, y, sigma):
    # return optimise_parameters_lm(f_np, param_syms, params_vals_org, X, y)
    return optimise_parameters(f_np, param_syms, params_vals_org, X, y, sigma)


def _predict(f_np, X, params):
    y_pred = evaluate_model(f_np, None, params, X)
    bad = (~np.isfinite(y_pred)) | (np.abs(y_pred) > 1e150)
    if bad.any():
        y_pred = np.where(bad, 1e150, y_pred)
    return y_pred


def calc_Err_in_sympy(
    expr_sym,
    var_syms,
    param_syms,
    params_vals_org,
    f_np,
    X,
    y,
    sigma,
    m_big,
    sigma_hat,
    B: int = 500,
    random_state: int | None = 42,
    mode: str = "rf",  # 'residual' or 'rf'
):
    if B < 2:
        raise ValueError("B must be ≥ 2 to estimate covariance (B-1 in denominator).")

    rng = np.random.default_rng(random_state)

    params_hat = _fit_params_lm(f_np, param_syms, params_vals_org, X, y, sigma)
    mu_hat = _predict(f_np, X, params_hat)  # fitted means
    resid = y - mu_hat
    n = len(y)

    if mode not in ("residual", "rf"):
        raise ValueError("mode must be 'residual' or 'rf'")

    y_star = np.empty((n, B))
    mu_star = np.empty_like(y_star)

    for b in range(B):
        if mode == "residual":
            eps_b = rng.choice(resid, size=n, replace=True)
            y_b = mu_hat + eps_b
        else:  # mode == 'rf'
            eps_b = rng.normal(0.0, sigma_hat, size=n)
            y_b = m_big + eps_b

        params_b = _fit_params_lm(f_np, param_syms, params_vals_org, X, y_b, sigma)
        mu_b = _predict(f_np, X, params_b)

        y_star[:, b], mu_star[:, b] = y_b, mu_b
        if b % 100 == 0:
            print(f"Completed bootstrap sample {b+1}/{B}")

    if mode == "rf":
        y_bar = m_big[:, None]
        denom = B
    else:
        y_bar = y_star.mean(axis=1, keepdims=True)
        denom = B - 1

    mu_bar = mu_star.mean(axis=1, keepdims=True)

    cov_i = ((mu_star - mu_bar) * (y_star - y_bar)).sum(
        axis=1
    ) / denom  # this is modified from Enfron eq. 2.15 by subtracting mu_bar

    err_app = np.mean((y - mu_hat) ** 2)
    optimism = 2.0 * cov_i.mean()
    Err = err_app + optimism

    return Err, err_app, optimism, params_hat


def _init_worker_bfgs(f_np, X, y, sigma, distribution):
    global _worker_f_np, _worker_X, _worker_y, _worker_sigma, _worker_distribution
    _worker_f_np = f_np
    _worker_X = X
    _worker_y = y
    _worker_sigma = sigma
    _worker_distribution = distribution


def _init_worker_lm(f_np, X, y):
    global _worker_f_np, _worker_X, _worker_y
    _worker_f_np = f_np
    _worker_X = X
    _worker_y = y


def _run_single_restart_bfgs(x0):
    f_np = _worker_f_np
    X = _worker_X
    y = _worker_y
    sigma = _worker_sigma
    distribution = _worker_distribution

    res = minimize(
        negative_log_likelihood,
        x0,
        args=(f_np, X, y, distribution, sigma),
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


def _run_single_restart_lm(x0):
    f_np = _worker_f_np
    X = _worker_X
    y = _worker_y

    def residuals(params):
        y_pred = evaluate_model(f_np, None, params, X)
        bad = (~np.isfinite(y_pred)) | (np.abs(y_pred) > 1e150)
        if bad.any():
            y_pred = np.where(bad, 1e12, y_pred)
        return y_pred - y

    res = least_squares(
        residuals,
        x0=x0,
        method="lm",
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
        max_nfev=4000,
    )

    theta_candidate = res.x
    y_pred = evaluate_model(f_np, None, theta_candidate, X)
    bad = (~np.isfinite(y_pred)) | (np.abs(y_pred) > 1e150)
    if bad.any():
        y_pred = np.where(bad, 1e12, y_pred)
    resid = y_pred - y
    sse = float(np.dot(resid, resid))

    return sse, theta_candidate


def optimise_parameters_lm(
    f_np,
    param_syms,
    params_vals_org,
    X,
    y,
    distribution: str = "gaussian",
):
    n_restarts = 100
    n_jobs = None

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

    global _worker_f_np, _worker_X, _worker_y

    try:
        try:
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = mp.get_context()

        with ctx.Pool(
            processes=n_jobs,
            initializer=_init_worker_lm,
            initargs=(f_np, X, y),
        ) as pool:
            results = pool.map(_run_single_restart_lm, x0_list)

        best_sse = np.inf
        best_theta = base_init.copy()

        for sse_candidate, theta_candidate in results:
            if sse_candidate < best_sse:
                best_sse = sse_candidate
                best_theta = theta_candidate

        return best_theta.tolist()

    finally:
        _worker_f_np = None
        _worker_X = None
        _worker_y = None


def negative_log_likelihood_gaussian(f_np, X, y, theta):
    """
    Computes the negative log-likelihood of the Gaussian model at the given parameters.
    """
    sigma2 = float(theta[-1])
    if sigma2 <= 0:
        return float("inf")

    Xcols = [X[:, i] for i in range(X.shape[1])]
    preds = f_np(*Xcols, *theta[:-1])
    resid = y - preds
    m = float(len(y))

    sse = np.dot(resid, resid)
    return 0.5 * (sse / sigma2 + m * math.log(2.0 * math.pi * sigma2))


def fisher_diag_gaussian(f_np, X, y, theta, eps=1e-6):
    """
    Compute the Fisher information diagonal for the Gaussian negative log-likelihood.
    """
    theta = np.asarray(theta, dtype=float)
    p = len(theta)
    fisher = np.empty(p, dtype=float)

    base_nll = negative_log_likelihood_gaussian(f_np, X, y, theta)

    for idx in range(p):
        theta_plus = theta.copy()
        theta_minus = theta.copy()

        theta_plus[idx] += eps
        theta_minus[idx] -= eps

        f_plus = negative_log_likelihood_gaussian(f_np, X, y, theta_plus)
        f_minus = negative_log_likelihood_gaussian(f_np, X, y, theta_minus)

        fisher[idx] = (f_plus + f_minus - 2.0 * base_nll) / (eps * eps)

    return fisher


def log_parameters_mdl(f_np, X, y, theta):
    fisher = fisher_diag_gaussian(f_np, X, y, theta)

    log_theta_sum = 0.0
    log_fisher_sum = 0.0
    significant_p_count = 0

    for v, f in zip(theta, fisher):
        if f <= 0:  # Avoid math domain errors for log(f) and sqrt(12/f)
            continue

        is_significant = abs(v / math.sqrt(12 / f)) >= 1

        if is_significant:
            log_theta_sum += math.log(abs(v))
            log_fisher_sum += math.log(f)
            significant_p_count += 1

    p = float(significant_p_count)

    return -(p / 2.0) * math.log(3) + 0.5 * log_fisher_sum + log_theta_sum


def model_selection_metrics(
    expr_sym,
    expr_sym_org,
    var_syms,
    f_np,
    f_np_org,
    params,
    params_vals_org,
    theta_full,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    y_clean_test,
    number_of_nodes,
    model_id,
    distribution="gaussian",
    param_syms=None,
    sigma=None,
):
    # N = len(y_train) + len(y_val) + len(y_test)
    n_nodes = number_of_nodes
    # n_params  = len(params)
    n_params = len(params_vals_org)
    k_penalty = n_params + (1 if distribution == "gaussian" else 0)

    nll_tr_org = negative_log_likelihood(
        [], f_np_org, X_train, y_train, distribution, sigma
    )
    # nll_val_org = negative_log_likelihood([], f_np_org, X_val, y_val, distribution, sigma)
    nll_te_org = negative_log_likelihood(
        [], f_np_org, X_test, y_clean_test, distribution, sigma
    )

    nll_tr = negative_log_likelihood(
        params, f_np, X_train, y_train, distribution, sigma
    )
    # nll_val = negative_log_likelihood(params, f_np, X_val, y_val, distribution, sigma)
    nll_te = negative_log_likelihood(
        params, f_np, X_test, y_clean_test, distribution, sigma
    )

    # sse_tr_org  = compute_SSE(y_train, f_np_org(*[X_train[:,i] for i in range(X_train.shape[1])]))
    # sse_val_org = compute_SSE(y_val, f_np_org(*[X_val[:,i] for i in range(X_val.shape[1])]))
    # sse_te_org  = compute_SSE(y_clean_test, f_np_org(*[X_test[:,i] for i in range(X_test.shape[1])]))

    # sse_tr  = compute_SSE(y_train, f_np(*([X_train[:,i] for i in range(X_train.shape[1])] + params)))
    # sse_val = compute_SSE(y_val,   f_np(*([X_val[:,i]   for i in range(X_val.shape[1])]+ params)))
    # sse_te  = compute_SSE(y_clean_test,  f_np(*([X_test[:,i]  for i in range(X_test.shape[1])]+ params)))

    mse_tr_org = compute_MSE(
        y_train, f_np_org(*[X_train[:, i] for i in range(X_train.shape[1])])
    )
    # mse_val_org = compute_MSE(y_val, f_np_org(*[X_val[:,i] for i in range(X_val.shape[1])]))
    mse_te_org = compute_MSE(
        y_clean_test, f_np_org(*[X_test[:, i] for i in range(X_test.shape[1])])
    )

    mse_tr = compute_MSE(
        y_train, f_np(*([X_train[:, i] for i in range(X_train.shape[1])] + params))
    )
    # mse_val = compute_MSE(y_val,   f_np(*([X_val[:,i]   for i in range(X_val.shape[1])]+ params)))
    mse_te = compute_MSE(
        y_clean_test,
        f_np(*([X_test[:, i] for i in range(X_test.shape[1])] + params)),
    )
    mse_te_noisy = compute_MSE(
        y_test, f_np(*([X_test[:, i] for i in range(X_test.shape[1])] + params))
    )

    # information criteria
    aic = 2 * k_penalty + 2 * nll_tr
    aicc = aic + 2 * k_penalty * (k_penalty + 1) / (len(y_train) - k_penalty - 1)
    bic = 2 * nll_tr + k_penalty * math.log(len(y_train))
    # aic_val = 2*k_penalty + 2*nll_val if nll_val is not None else None
    # aicc_val = aic_val + 2*k_penalty*(k_penalty+1)/(len(y_val)-k_penalty-1)
    # bic_val = (2*nll_val + k_penalty*math.log(len(y_val))) if nll_val is not None else None

    # ---------- constants for Eq. (6) ------------------------------
    # n_unique_ops = len(_unique_ops(expr_sym))
    # const_vals   = _numeric_constants(expr_sym)
    # ##### MDL_train
    log_functional = log_functional_spec(
        expr_with_params=expr_sym, expr_original=expr_sym_org, n_nodes=n_nodes
    )
    fisher_diag = fisher_diag_gaussian(f_np, X_train, y_train, theta_full)
    log_parameters = log_parameters_mdl(f_np, X_train, y_train, theta_full)

    mdl = nll_tr + log_functional + log_parameters

    # ##### MDL_val
    # fisher_diag_val = fisher_diag_gaussian(f_np,X_val, y_val, theta_full)
    # log_parameters_val = log_parameters_mdl(f_np,X_val, y_val, theta_full)

    # mdl_val = nll_val_org + log_functional + log_parameters_val

    # # --- MDL (freq & lattice)
    # func_complexity  = n_nodes * math.log(2)                       # nats
    # param_complexity = 0.5 * k_penalty * math.log(len(y_train))    # Rissanen

    # mdl_freq   = nll_tr + 0.9*func_complexity + param_complexity  # To be implemented
    # mdl_lattice= mdl # To be implemented

    return {
        "Index": model_id,
        "Expression": expr_sym_org,
        "Expression_sym": expr_sym,
        "Number_of_nodes": n_nodes,
        "Number_of_parameters": n_params,
        "Parameters": params_vals_org,
        "Parameters_opt": params,
        # "SSE_train_orig": sse_tr_org,
        # "SSE_val_orig":   sse_val_org,
        # "SSE_test_orig":  sse_te_org,
        # "SSE_train_opt": sse_tr,
        # "SSE_val_opt":   sse_val,
        # "SSE_test_opt":  sse_te,
        "MSE_train_orig": mse_tr_org,
        # "MSE_val_orig":   mse_val_org,
        "MSE_test_orig": mse_te_org,
        "MSE_train_opt": mse_tr,
        # "MSE_val_opt":   mse_val,
        "MSE_test_opt": mse_te,
        "MSE_test_opt_noisy": mse_te_noisy,
        "AIC": aic,
        # "AIC_val": aic_val,
        "AICc": aicc,
        # "AICc_val": aicc_val,
        "BIC": bic,
        # "BIC_val": bic_val,
        "MDL": mdl,
        # "MDL_freq": mdl_freq,
        # "MDL_lattice": mdl_lattice,
        # "MDL_val": mdl_val,
        "NegLogLikelihood_train_opt": nll_tr,
        "NegLogLikelihood_train_orig": nll_tr_org,
        # "NegLogLikelihood_val_opt":   nll_val,
        # "NegLogLikelihood_val_orig":   nll_val_org,
        "NegLogLikelihood_test_opt": nll_te,
        "NegLogLikelihood_test_orig": nll_te_org,
        "LogFunctional": log_functional,
        "LogParameters": log_parameters,
        "Fisher": fisher_diag,
    }


def compute_selection_metrics(
    fitted_models_Err,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    y_clean_test,
    x_cols,
    sigma,
    n_nodes,
):
    base_metrics = []
    Err_metrics = []
    print(f"#train: {len(y_train)}, #val: {len(y_val)}, #test: {len(y_test)}")
    for (
        idx,
        expr_sym,
        expr_sym_org,
        expr_fitted,
        f_np_fitted,
        f_np_org,
        var_syms,
        param_syms,
        param_vals_opt,
        params_vals_org,
    ) in fitted_models_Err:
        Xtr = reorder_X(X_train, var_syms, x_cols)
        Xva = reorder_X(X_val, var_syms, x_cols)
        Xte = reorder_X(X_test, var_syms, x_cols)
        if np.isnan(Xva.any()) or np.isnan(Xte.any()):
            print("Khata", idx)
            print(X)
        f_full = sp.lambdify(
            [*var_syms, *param_syms], expr_sym, modules=["numpy", CUSTOM_NUMPY]
        )
        theta_full = np.array(list(param_vals_opt) + [float(sigma) ** 2], dtype=float)
        if n_nodes is not None:
            number_of_nodes = n_nodes[idx]
        else:
            number_of_nodes = count_number_of_nodes(expr_sym_org, var_syms)
        metrics = model_selection_metrics(
            expr_sym,
            expr_sym_org,
            var_syms,
            f_full,
            f_np_org,
            params=param_vals_opt,
            params_vals_org=params_vals_org,
            theta_full=theta_full,
            X_train=Xtr,
            y_train=y_train,
            X_val=Xva,
            y_val=y_val,
            X_test=Xte,
            y_test=y_test,
            y_clean_test=y_clean_test,
            number_of_nodes=number_of_nodes,
            model_id=idx,
            distribution="gaussian",
            param_syms=param_syms,
            sigma=sigma,
        )

        base_metrics.append(metrics)

    print("##### Calculation of AIC, AICc, BIC, MDL done #####")
    print(
        "######################################################### Calculating Err_in"
    )

    for (
        idx,
        expr_sym,
        expr_sym_org,
        expr_fitted,
        f_np_fitted,
        f_np_org,
        var_syms,
        param_syms,
        param_vals_opt,
        params_vals_org,
    ) in fitted_models_Err:
        print(f"Calculating Err_in for model {idx}")
        Xtr = reorder_X(X_train, var_syms, x_cols)
        Xva = reorder_X(X_val, var_syms, x_cols)
        Xte = reorder_X(X_test, var_syms, x_cols)
        f_full = sp.lambdify(
            [*var_syms, *param_syms], expr_sym, modules=["numpy", CUSTOM_NUMPY]
        )

        random_state = 42
        rf = RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
            bootstrap=True,
            oob_score=True,
        )
        rf.fit(Xtr, y_train)
        m_big = rf.predict(Xtr)  # center for the generator
        y_hat_oob = rf.oob_prediction_
        mask = ~np.isnan(y_hat_oob)
        sigma_hat = float(np.sqrt(np.mean((y_train[mask] - y_hat_oob[mask]) ** 2)))

        Err, err_app, optimism, params_hat = calc_Err_in_sympy(
            expr_sym,
            var_syms,
            param_syms,
            params_vals_org,
            f_full,
            Xtr,
            y_train,
            sigma,
            m_big,
            sigma_hat,
            random_state=random_state,
            B=200,
            mode="rf",
        )

        Err_metrics.append(
            {
                "Err_in": Err,
                "optimism": optimism,
                "err_app": err_app,
                "Index": idx,
            }
        )

    return base_metrics, Err_metrics, theta_full


for k, _ in models_path.items():
    print(f"***** Function {k} started *****")

    function_name = k  # "f1" # Choose function f1-f7
    mean_values_of_the_models = models_path[function_name][:-6] + "txt"

    operon_file = Path(models_path[function_name])

    # ###################### Load separate data points
    csv_path_train_val = Path(
        "data/" + data_points_path[function_name] + "_train_val.csv"
    )
    print(csv_path_train_val)
    csv_path_test = Path("data/" + data_points_path[function_name] + "_test.csv")
    print(csv_path_test)
    N = 100  # Number of rows for training
    N_val = 20  # Validation is not used in this code, but kept for consistency

    df_train_val = pd.read_csv(csv_path_train_val)
    df_test = pd.read_csv(csv_path_test)

    x_cols = [c for c in df_train_val.columns if c.startswith("x")]
    X_train_val = df_train_val[x_cols].to_numpy(float)
    y_train_val = df_train_val["y"].to_numpy(float)
    y_clean_train_val = df_train_val["y_clean"].to_numpy(float)

    X_test = df_test[x_cols].to_numpy(float)
    y_test = df_test["y"].to_numpy(float)
    y_clean_test = df_test["y_clean"].to_numpy(float)

    X_train, y_train, y_clean_train = (
        X_train_val[:N],
        y_train_val[:N],
        y_clean_train_val[:N],
    )
    X_val, y_val, y_clean_val = (
        X_train_val[N - N_val :],
        y_train_val[N - N_val :],
        y_clean_train_val[N - N_val :],
    )

    # Load the sigma_err value
    sigma_err = df_train_val["sigma"][
        0
    ]  # sigma is constant and same for all rows across each dataset

    FLOAT_MAX = np.finfo(np.float64).max
    CLIP_RESID = 1e12

    # Parallel helpers
    # ------------------------------------------------------------------

    _worker_f_np = None
    _worker_X = None
    _worker_y = None
    _worker_sigma = None
    _worker_distribution = None

    # ────────────────────────────────────────────────────────────────
    # Main loops for model loading
    # ────────────────────────────────────────────────────────────────

    print("############################## Loading for Err_in")

    fitted_models_Err = []
    results_Err = []

    convert_consts_Err = False
    for idx, (
        expr_sym,
        expr_sym_org,
        var_syms,
        param_syms,
        params_vals_org,
        f_np,
        f_np_org,
    ) in enumerate(load_operon(operon_file, convert_consts_Err, parser="Err")):
        X = reorder_X(X_train, var_syms, x_cols)
        param_vals_opt = optimise_parameters(
            f_np,
            param_syms,
            params_vals_org,
            X,
            y_train,
            sigma_err,
            distribution="gaussian",
        )

        if param_syms:
            subst_map = dict(zip(param_syms, param_vals_opt))
            expr_fitted = expr_sym.subs(subst_map).evalf()
        else:
            expr_fitted = expr_sym

        f_np_fitted = sp.lambdify(
            var_syms, expr_fitted, modules=["numpy", CUSTOM_NUMPY]
        )

        fitted_models_Err.append(
            (
                idx,
                expr_sym,
                expr_sym_org,
                expr_fitted,
                f_np_fitted,
                f_np_org,
                var_syms,
                param_syms,
                param_vals_opt,
                params_vals_org,
            )
        )

        y_pred = f_np_fitted(*[X[:, i] for i in range(X.shape[1])])

        if not np.all(np.isfinite(y_pred)):
            sse = np.inf
        else:
            err = np.clip(y_train - y_pred, -CLIP_RESID, CLIP_RESID)
            sse = float(np.dot(err, err))

        X_evaluation = reorder_X(X_train_val, var_syms, x_cols)
        eval_tmp = evaluate_model(f_np_org, None, [], X_evaluation)
        results_Err.append((idx, sse, str(expr_fitted), np.mean(eval_tmp)))
        if idx % 10 == 0:
            print(f"Processed {idx} models for Err_in")

    base_metrics, Err_metrics, theta_full = compute_selection_metrics(
        fitted_models_Err,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        y_clean_test,
        x_cols,
        sigma_err,
        n_nodes=None,
    )

    df_base_metrics = pd.DataFrame(base_metrics)
    df_Err_metrics = pd.DataFrame(Err_metrics)
    df_metrics = df_base_metrics.merge(df_Err_metrics, on="Index", how="inner")
    df_metrics.to_csv(f"BFGS_cluster_results_{operon_file.stem}.csv")
    print(f"***** Function {k} finished *****")
