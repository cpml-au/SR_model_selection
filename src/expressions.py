import re
import numpy as np
import sympy as sp
from typing import Iterator, Tuple, List, Callable
from pathlib import Path
from sympy.parsing.sympy_parser import parse_expr, standard_transformations
from sympy.core import evaluate

from ops import protected_div, protected_pow, protected_sqrt, protected_exp

CUSTOM_NUMPY = {
    "Max": np.maximum,
    "Min": np.minimum,
    "protected_div": protected_div,
    "protected_pow": protected_pow,
    "protected_sqrt": protected_sqrt,
    "protected_exp": protected_exp,
}


def parse_all_numeric_expression(expr_str):
    var_names = sorted(
        {m.group() for m in re.finditer(r"x\d+", expr_str)},
        key=lambda s: int(s[1:]),
    )
    t_names_in_expr = sorted(
        {m.group() for m in re.finditer(r"t\d+", expr_str)},
        key=lambda s: int(s[1:]),
    )

    sym_dict = {n: sp.symbols(n) for n in (var_names + t_names_in_expr)}

    SYM_FUNCS = {
        "protected_div": sp.Function("protected_div"),
        "protected_pow": sp.Function("protected_pow"),
        "protected_sqrt": sp.Function("protected_sqrt"),
        "protected_exp": sp.Function("protected_exp"),
    }
    NUMPY_FUNCS = {
        "Max": np.maximum,
        "Min": np.minimum,
        "protected_div": protected_div,
        "protected_pow": protected_pow,
        "protected_sqrt": protected_sqrt,
        "protected_exp": protected_exp,
    }

    with evaluate(False):
        expr_sym = parse_expr(
            expr_str,
            local_dict={**sym_dict, **SYM_FUNCS, "Max": sp.Max, "Min": sp.Min},
            transformations=standard_transformations,
            evaluate=False,
        )
    expr_sym_org = expr_sym

    def _extract_t_index(s: str):
        m = re.fullmatch(r"t(\d+)", s)
        return int(m.group(1)) if m else None

    max_t = max(
        (
            idx
            for s in t_names_in_expr
            for idx in [_extract_t_index(s)]
            if idx is not None
        ),
        default=0,
    )
    next_idx = max_t + 1

    created_syms_vals = []
    from sympy import S

    def new_t(init_val: float):
        nonlocal next_idx, created_syms_vals
        t = sp.symbols(f"t{next_idx}")
        created_syms_vals.append((t, float(init_val)))
        next_idx += 1
        return t

    def _collapse_neg_one_times_number(node):
        if isinstance(node, sp.Mul):
            args = list(node.args)
            if (
                S.NegativeOne in args
                and any(isinstance(a, sp.Number) for a in args)
                and not any(isinstance(a, sp.Symbol) for a in args)
            ):
                num = S.One
                sign = 1
                others = []
                for a in args:
                    if a == S.NegativeOne:
                        sign *= -1
                    elif isinstance(a, sp.Number):
                        num = sp.Mul(num, a, evaluate=True)
                    else:
                        others.append(a)
                coeff = -num if sign < 0 else num
                if others:
                    return sp.Mul(coeff, *others, evaluate=False)
                else:
                    return coeff
        return node

    with evaluate(False):
        expr_sym = expr_sym.replace(
            lambda n: isinstance(n, sp.Mul), _collapse_neg_one_times_number
        )

    def _minus_one_times_single_factor(node):
        if isinstance(node, sp.Mul):
            args = list(node.args)
            if S.NegativeOne in args:
                others = [a for a in args if a is not S.NegativeOne]
                if len(others) == 1 and not isinstance(others[0], sp.Number):
                    t = new_t(-1.0)
                    return sp.Mul(t, others[0], evaluate=False)
        return node

    with evaluate(False):
        expr_sym = expr_sym.replace(
            lambda n: isinstance(n, sp.Mul), _minus_one_times_single_factor
        )

    with evaluate(False):
        expr_sym = expr_sym.replace(
            lambda n: isinstance(n, sp.Number),
            lambda n: n if n in (S.Zero, S.One) else new_t(float(n)),
        )

    var_symbols = [sym_dict[n] for n in var_names]
    existing_t_params = [sp.symbols(n) for n in t_names_in_expr]
    created_t_params = [t for (t, _) in created_syms_vals]
    created_t_values = [v for (_, v) in created_syms_vals]

    param_symbols = existing_t_params + created_t_params
    param_values = [None] * len(existing_t_params) + created_t_values

    f_np = sp.lambdify(
        [*var_symbols, *param_symbols], expr_sym, modules=["numpy", NUMPY_FUNCS]
    )
    f_np_org = sp.lambdify(var_symbols, expr_sym_org, modules=["numpy", NUMPY_FUNCS])

    return (
        expr_sym,
        expr_sym_org,
        var_symbols,
        param_symbols,
        param_values,
        f_np,
        f_np_org,
    )


def load_operon(
    path: str | Path, convert_consts: bool = True, parser: str = "Fab"
) -> Iterator[Tuple[sp.Expr, List[sp.Symbol], List[sp.Symbol], Callable]]:
    path = Path(path).expanduser()
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                if parser == "Err":
                    yield parse_all_numeric_expression(line)
                else:
                    print("Please define parser correctly.")


def count_number_of_nodes(expr, syms):
    constants_terminals = [
        a for a in sp.preorder_traversal(expr) if a.is_Number
    ]  # and a not in (-1,1)]
    operators = expr.count_ops(visual=False)
    variable_terminals = syms
    return len(constants_terminals) + operators + len(variable_terminals)
