import numpy as np

from sklearn.ensemble import RandomForestRegressor

from optimize import _fit_params_lm, _predict


def calc_Err_in_sympy(
    param_syms,
    params_vals_org,
    f_np,
    X,
    y,
    sigma,
    B: int = 500,
    random_state: int | None = 42,
    mode: str = "rf",
):
    if B < 2:
        raise ValueError("B must be ≥ 2 to estimate covariance (B-1 in denominator).")

    rng = np.random.default_rng(random_state)

    params_hat = _fit_params_lm(f_np, param_syms, params_vals_org, X, y, sigma)
    mu_hat = _predict(f_np, X, params_hat)
    resid = y - mu_hat
    n = len(y)

    if mode not in ("residual", "rf"):
        raise ValueError("mode must be 'residual' or 'rf'")

    y_star = np.empty((n, B))
    mu_star = np.empty_like(y_star)

    m_big = None
    sigma_hat = None
    if mode == "rf":
        rf = RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
            bootstrap=True,
            oob_score=True,
        )
        rf.fit(X, y)
        m_big = rf.predict(X)
        y_hat_oob = rf.oob_prediction_
        mask = ~np.isnan(y_hat_oob)
        sigma_hat = float(np.sqrt(np.mean((y[mask] - y_hat_oob[mask]) ** 2)))

    for b in range(B):
        if mode == "residual":
            eps_b = rng.choice(resid, size=n, replace=True)
            y_b = mu_hat + eps_b
        else:
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

    cov_i = ((mu_star - mu_bar) * (y_star - y_bar)).sum(axis=1) / denom

    err_app = np.mean((y - mu_hat) ** 2)
    optimism = 2.0 * cov_i.mean()
    Err = err_app + optimism

    return Err, err_app, optimism, params_hat
