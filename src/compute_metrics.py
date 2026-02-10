import sys
import numpy as np
import sympy as sp
from pathlib import Path
import pandas as pd
from expressions import (
    CUSTOM_NUMPY,
    load_operon,
    count_number_of_nodes,
)
from methods.aic import compute_aic
from methods.aicc import compute_aicc
from methods.bic import compute_bic
from methods.mdl import compute_mdl
from methods.errin import calc_Err_in_sympy
from optimize import optimise_parameters

np.random.seed(42)


n_points = 10100
n_level = 0.1
models_path = {
    "f1": "functions/f1_100_10.operon",
    "f2": "functions/f2_100_2.operon",
    "f3": "functions/f3_100_1.operon",
    "f4": "functions/f4_100_2.operon",
    "f5": "functions/f5_100_3.operon",
    "f6": "functions/f6_100_2.operon",
    "f7": "functions/f7_100_2.operon",
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


def evaluate_model(f_np, param_values, X):
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

    clip = 1e150
    err = np.nan_to_num(err, nan=clip, posinf=clip, neginf=-clip)
    err = np.clip(err, -clip, clip)
    return float(np.dot(err, err))


def compute_MSE(y_true, y_pred):
    return compute_SSE(y_true, y_pred) / len(y_true)


def reorder_X(X, var_symbols, all_columns):
    """Return a view of X whose columns match the order of var_symbols."""
    idx = [all_columns.index(sym.name) for sym in var_symbols]
    return X[:, idx]


def negative_log_likelihood(param_values, f_np, X, y, distribution, sigma):

    y_pred = evaluate_model(f_np, param_values, X)

    if (not np.all(np.isfinite(y_pred))) or np.any(np.abs(y_pred) > 1e150):
        return 1e40

    y_true = np.asarray(y, dtype=float)
    N = len(y_true)

    if distribution == "gaussian":

        sse = compute_SSE(y_true=y_true, y_pred=y_pred)
        sse = max(sse, 1e-12)
        if sigma is None:
            return 0.5 * N * (np.log(2 * np.pi) + 1.0 + np.log(sse / N))
        else:

            sigma2 = float(sigma) ** 2
            return 0.5 * N * np.log(2 * np.pi * sigma2) + 0.5 * sse / sigma2

    elif distribution == "bernoulli":
        p = 1.0 / (1.0 + np.exp(-y_pred))
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return -np.sum(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    elif distribution == "poisson":
        lam = np.exp(np.clip(y_pred, -50, 50))
        return np.sum(lam - y_true * np.log(lam.clip(1e-12)))

    raise ValueError("unknown distribution")


def model_selection_metrics(
    expr_sym,
    expr_sym_org,
    f_np,
    f_np_org,
    params,
    params_vals_org,
    param_syms,
    theta_full,
    X_train,
    y_train,
    X_test,
    y_test,
    y_clean_test,
    number_of_nodes,
    model_id,
    distribution="gaussian",
    sigma=None,
):

    n_nodes = number_of_nodes

    n_params = len(params_vals_org)
    k_penalty = n_params + (1 if distribution == "gaussian" else 0)

    random_state = 42
    nll_tr_org = negative_log_likelihood(
        [], f_np_org, X_train, y_train, distribution, sigma
    )

    nll_te_org = negative_log_likelihood(
        [], f_np_org, X_test, y_clean_test, distribution, sigma
    )

    nll_tr = negative_log_likelihood(
        params, f_np, X_train, y_train, distribution, sigma
    )

    nll_te = negative_log_likelihood(
        params, f_np, X_test, y_clean_test, distribution, sigma
    )

    mse_tr_org = compute_MSE(
        y_train, f_np_org(*[X_train[:, i] for i in range(X_train.shape[1])])
    )

    mse_te_org = compute_MSE(
        y_clean_test, f_np_org(*[X_test[:, i] for i in range(X_test.shape[1])])
    )

    mse_tr = compute_MSE(
        y_train, f_np(*([X_train[:, i] for i in range(X_train.shape[1])] + params))
    )

    mse_te = compute_MSE(
        y_clean_test,
        f_np(*([X_test[:, i] for i in range(X_test.shape[1])] + params)),
    )
    mse_te_noisy = compute_MSE(
        y_test, f_np(*([X_test[:, i] for i in range(X_test.shape[1])] + params))
    )

    Err, err_app, optimism, params_hat = calc_Err_in_sympy(
        param_syms,
        params_vals_org,
        f_np,
        X_train,
        y_train,
        sigma,
        random_state=random_state,
        B=200,
        mode="rf",
    )

    aic = compute_aic(k_penalty, nll_tr)
    aicc = compute_aicc(aic, k_penalty, len(y_train))
    bic = compute_bic(k_penalty, nll_tr, len(y_train))

    mdl = compute_mdl(
        nll=nll_tr,
        expr_sym=expr_sym,
        expr_sym_org=expr_sym_org,
        n_nodes=n_nodes,
        f_np=f_np,
        X_train=X_train,
        y_train=y_train,
        theta_full=theta_full,
    )

    return {
        "Index": model_id,
        "Expression": expr_sym_org,
        "Expression_sym": expr_sym,
        "Number_of_nodes": n_nodes,
        "Number_of_parameters": n_params,
        "Parameters": params_vals_org,
        "Parameters_opt": params,
        "MSE_train_orig": mse_tr_org,
        "MSE_test_orig": mse_te_org,
        "MSE_train_opt": mse_tr,
        "MSE_test_opt": mse_te,
        "MSE_test_opt_noisy": mse_te_noisy,
        "AIC": aic,
        "AICc": aicc,
        "BIC": bic,
        "MDL": mdl,
        "Err_in": Err,
        "optimism": optimism,
        "err_app": err_app,
        "NegLogLikelihood_train_opt": nll_tr,
        "NegLogLikelihood_train_orig": nll_tr_org,
        "NegLogLikelihood_test_opt": nll_te,
        "NegLogLikelihood_test_orig": nll_te_org,
    }


def compute_selection_metrics(
    fitted_models_Err,
    X_train,
    y_train,
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
        Xte = reorder_X(X_test, var_syms, x_cols)
        if np.isnan(Xte.any()):
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
            f_full,
            f_np_org,
            params=param_vals_opt,
            params_vals_org=params_vals_org,
            param_syms=param_syms,
            theta_full=theta_full,
            X_train=Xtr,
            y_train=y_train,
            X_test=Xte,
            y_test=y_test,
            y_clean_test=y_clean_test,
            number_of_nodes=number_of_nodes,
            model_id=idx,
            distribution="gaussian",
            sigma=sigma,
        )

        base_metrics.append(metrics)

    return base_metrics, theta_full


for k, _ in models_path.items():
    print(f"***** Function {k} started *****")

    function_name = k
    mean_values_of_the_models = models_path[function_name][:-6] + "txt"

    operon_file = Path(models_path[function_name])

    csv_path_train_val = Path(
        "data/" + data_points_path[function_name] + "_train_val.csv"
    )
    print(csv_path_train_val)
    csv_path_test = Path("data/" + data_points_path[function_name] + "_test.csv")
    print(csv_path_test)
    N = 100
    N_val = 20

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

    sigma_err = df_train_val["sigma"][0]

    FLOAT_MAX = np.finfo(np.float64).max
    CLIP_RESID = 1e12

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
        eval_tmp = evaluate_model(f_np_org, [], X_evaluation)
        results_Err.append((idx, sse, str(expr_fitted), np.mean(eval_tmp)))
        if idx % 10 == 0:
            print(f"Processed {idx} models for Err_in")

    base_metrics, theta_full = compute_selection_metrics(
        fitted_models_Err,
        X_train,
        y_train,
        X_test,
        y_test,
        y_clean_test,
        x_cols,
        sigma_err,
        n_nodes=None,
    )

    df_base_metrics = pd.DataFrame(base_metrics)
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    df_base_metrics.to_csv(
        results_dir / f"model_selection_methods_{operon_file.stem}.csv"
    )
    print(f"***** Function {k} finished *****")
