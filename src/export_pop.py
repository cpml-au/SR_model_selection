import random
import argparse
import pathlib
import pandas as pd
import numpy as np
from deap import gp
from tree_mutation import (
    ground_tree_generator,
    ground_truth_perturbations,
    pset_definition,
)


def parse_deap_expression(expr: str) -> str:
    expr = expr.strip()
    if "(" not in expr:
        return expr

    open_paren = expr.index("(")
    close_paren = expr.rindex(")")
    func = expr[:open_paren].strip()
    raw_args = expr[open_paren + 1 : close_paren]

    depth = 0
    arg, args = [], []
    for ch in raw_args:
        if ch == "," and depth == 0:
            args.append("".join(arg).strip())
            arg = []
        else:
            depth += ch == "("
            depth -= ch == ")"
            arg.append(ch)
    if arg:
        args.append("".join(arg).strip())

    _binary_ops = {
        "add": "+",
        "sub": "-",
        "subtract": "-",
        "mul": "*",
        "multiply": "*",
        "div": "/",
        "divide": "/",
        "pow": "**",
        "max": "Max",
        "min": "Min",
    }
    _unary_ops = {
        "negative": "-",
        "neg": "-",
        "u_minus": "-",
        "positive": "+",
        "pos": "+",
    }
    _unary_fun = {
        "sin",
        "cos",
        "tan",
        "exp",
        "log",
        "sqrt",
        "abs",
        "asin",
        "acos",
        "atan",
    }

    args = [parse_deap_expression(a) for a in args]

    if func in _binary_ops and len(args) == 2:
        op = _binary_ops[func]
        return f"({args[0]} {op} {args[1]})"

    if func in _unary_ops and len(args) == 1:
        op = _unary_ops[func]
        return f"({op}{args[0]})"

    if func in _unary_fun and len(args) == 1:
        return f"{func}({args[0]})"

    if func == "protected_pow" and len(args) == 2:
        return f"protected_pow({args[0]}, {args[1]})"

    if func == "protected_div" and len(args) == 2:
        return f"protected_div({args[0]}, {args[1]})"

    if func == "protected_sqrt" and len(args) == 1:
        return f"protected_sqrt({args[0]})"

    if func == "protected_exp":
        return f"protected_exp({args[0]})"

    raise ValueError(f"Unsupported function: {func}")


def export_operon(
    population, pset: gp.PrimitiveSet, path: str | pathlib.Path
) -> pathlib.Path:
    path = pathlib.Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as fh:
        for ind in population:
            fh.write(parse_deap_expression(str(ind)) + "\n")

    return path


def export_other_info(population, path: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as fh:
        for ind in population:
            fh.write(str(ind) + "\n")

    return path


data_points_path = {
    "f1": "friedman_10100_noise-0.1",
    "f2": "kotanchek_10100_noise-0.1",
    "f3": "salustowicz_10100_noise-0.1",
    "f4": "salustowicz2d_10100_noise-0.1",
    "f5": "ratpol3d_10100_noise-0.1",
    "f6": "ratpol2d_10100_noise-0.1",
    "f7": "ripple_10100_noise-0.1",
}


def main(n_mutations, n_features, function_name):
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    print(
        f"################# Running with n_mutations={n_mutations}, n_features={n_features}, seed={SEED}"
    )
    pset, toolbox = pset_definition(n_features)
    ground_tree = ground_tree_generator(pset, function_name=function_name)
    csv_path_train_val = pathlib.Path(
        "data/" + data_points_path[function_name] + "_train_val.csv"
    )
    df_train_val = pd.read_csv(csv_path_train_val)
    x_cols = [c for c in df_train_val.columns if c.startswith("x")]
    X_train_val = df_train_val[x_cols].to_numpy(float)
    y_train_val = df_train_val["y"].to_numpy(float)
    populations = ground_truth_perturbations(
        pset,
        toolbox,
        ground_tree,
        X_train=X_train_val,
        Y_train=y_train_val,
        n_mutations=n_mutations,
    )
    populations_list = [inds["tree"] for inds in populations]
    populations_values_list = [
        (inds["mean_train_pred"], inds["num_of_params"], str(inds["tree"]))
        for inds in populations
    ]
    print("Expression generation finished")
    functions_dir = pathlib.Path("functions")
    functions_dir.mkdir(parents=True, exist_ok=True)
    results_dir = pathlib.Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    path = export_operon(
        populations_list,
        pset,
        functions_dir / f"{function_name}_{n_mutations}_{n_features}.operon",
    )
    path_info = export_other_info(
        populations_values_list,
        results_dir / f"{function_name}_{n_mutations}_{n_features}.txt",
    )
    print(f"Expressions saved in: {path}")
    print(f"Expressions saved in: {path_info}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run perturbations on the selected ground truth tree."
    )
    parser.add_argument(
        "--n_m", type=int, default=10, help="Number of perturbed tree to generate."
    )
    parser.add_argument(
        "--n_f", type=int, default=10, help="Number of features in the problem set."
    )
    parser.add_argument(
        "--f_n",
        type=str,
        default="friedman",
        help="Name of the ground truth function(f1-f7).",
    )

    args = parser.parse_args()

    main(n_mutations=args.n_m, n_features=args.n_f, function_name=args.f_n)
