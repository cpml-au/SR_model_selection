import math
import numpy as np
import sympy as sp


def _has_sub(expr: sp.Expr) -> bool:
    """Detect any subtraction anywhere: a - b with any b."""
    for n in sp.preorder_traversal(expr):
        if isinstance(n, sp.Add):
            for arg in n.args:
                if isinstance(arg, sp.Mul):
                    coeff, _ = arg.as_coeff_Mul()
                    if coeff == -1:
                        return True
    return False


def _has_any_param(expr_with_params: sp.Expr, param_prefixes=("t",)) -> bool:
    for n in sp.preorder_traversal(expr_with_params):
        if isinstance(n, sp.Symbol):
            if any(n.name.startswith(p) for p in param_prefixes):
                return True
    return False


def _tokens_unique_spec(
    expr_with_params: sp.Expr, expr_original: sp.Expr, param_prefixes=("t",)
) -> set[str]:
    toks = set()
    have_param = _has_any_param(expr_with_params, param_prefixes)
    # have_const  = has_any_int_constant(expr_original)
    have_sub = _has_sub(expr_with_params)

    for n in sp.preorder_traversal(expr_with_params):
        if isinstance(n, sp.Symbol):
            name = n.name
            if not any(name.startswith(p) for p in param_prefixes):
                toks.add(name)
        elif isinstance(n, sp.Function):
            toks.add(type(n).__name__.capitalize())
        elif isinstance(n, sp.Mul):
            toks.add("Mul")
        elif isinstance(n, sp.Add):
            toks.add("Add")
        elif isinstance(n, sp.Pow):
            toks.add("Pow")

    if have_param:
        toks.add("Param")
    # if have_const:
    #     toks.add("Const")
    if have_sub:
        toks.add("Sub")

    return toks


def _ints_by_occurrence(expr_original: sp.Expr) -> list[int]:
    ints = []
    for n in sp.preorder_traversal(expr_original):
        if isinstance(n, sp.Integer):
            v = int(n)
            if v not in (0, 1, -1):
                ints.append(v)
        elif isinstance(n, sp.Float):
            f = float(n)
            if f.is_integer():
                v = int(round(f))
                if v not in (0, 1, -1):
                    ints.append(v)
    return ints


def log_functional_spec(
    expr_with_params: sp.Expr, expr_original: sp.Expr, n_nodes: int
) -> float:
    unique_tokens = len(_tokens_unique_spec(expr_with_params, expr_original))
    first = n_nodes * math.log(max(unique_tokens, 1))

    ints = _ints_by_occurrence(expr_original)
    second = sum(math.log(abs(v)) for v in ints)
    third = len(ints) * math.log(2.0)

    return first + second + third


def fisher_diag_gaussian(f_np, X, y, theta, eps=1e-6):
    """
    Compute the Fisher information diagonal for the Gaussian negative log-likelihood.
    """
    theta = np.asarray(theta, dtype=float)
    diag = np.empty_like(theta, dtype=float)

    base_nll = negative_log_likelihood_gaussian(f_np, X, y, theta)

    for i in range(len(theta)):
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[i] += eps
        theta_minus[i] -= eps
        f_plus = negative_log_likelihood_gaussian(f_np, X, y, theta_plus)
        f_minus = negative_log_likelihood_gaussian(f_np, X, y, theta_minus)
        diag[i] = (f_plus - 2 * base_nll + f_minus) / (eps ** 2)

    return diag


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


def log_parameters_mdl(f_np, X, y, theta):
    fisher = fisher_diag_gaussian(f_np, X, y, theta)
    fisher = np.clip(fisher, 1e-12, None)
    log_det = np.sum(np.log(fisher))
    return 0.5 * log_det


def compute_mdl(
    *,
    nll: float,
    expr_sym=None,
    expr_sym_org=None,
    n_nodes: int | None = None,
    f_np=None,
    X_train=None,
    y_train=None,
    theta_full=None,
) -> float:
    if expr_sym is None or expr_sym_org is None or n_nodes is None:
        raise ValueError(
            "expr_sym, expr_sym_org, and n_nodes are required to compute MDL."
        )
    if f_np is None or X_train is None or y_train is None or theta_full is None:
        raise ValueError(
            "f_np, X_train, y_train, and theta_full are required to compute MDL."
        )
    log_functional = log_functional_spec(
        expr_with_params=expr_sym, expr_original=expr_sym_org, n_nodes=n_nodes
    )
    log_parameters = log_parameters_mdl(f_np, X_train, y_train, theta_full)
    return nll + log_functional + log_parameters
