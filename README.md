# Symbolic Regression Model Selection

This repository contains the code used to generate results for the paper "A Comparative Study of Model Selection Criteria for Symbolic Regression".

## 1) Dataset Construction

```bash
cd src
python generate_datasets.py
```

Datasets are written to `src/data`.

## 2) Symbolic Model Generation

Generate perturbed models from a ground-truth function:

```bash
python export_pop.py --n_m <num_mutations> --n_f <num_features> --f_n <function_name>
```

- `--n_m`: number of perturbed models to generate
- `--n_f`: number of features (e.g., `f1` uses 10 features)
- `--f_n`: ground-truth function name (`f1`–`f7`)

Example (100 models for `f1`):

```bash
python export_pop.py --n_m 100 --n_f 10 --f_n f1
```

To generate all functions:

```bash
chmod +x export_pop_all.sh
./export_pop_all.sh
```

Outputs are `.operon` files named `<function>_<mutations>_<features>.operon` in `src/functions`.

## 3) Compute Selection Metrics

Compute metrics for one or more functions:

```bash
python compute_metrics.py [f1 f2 ...]
```

If no function names are provided, metrics are computed for all seven functions. Results are written to `results/` with names like `model_selection_methods_f1_100_10.csv`.

Example:

```bash
python compute_metrics.py f1
```

## 4) Evaluate Metric Performance

```bash
python eval_modelsel.py <k> <results_csv> <test_column>
```

- `<k>`: number of top-ranked models to evaluate
- `<results_csv>`: CSV from the previous step
- `<test_column>`: ground-truth metric (e.g., `MSE_test_opt`)

Example:

```bash
python eval_modelsel.py 50 results/model_selection_methods_f1_100_10.csv MSE_test_opt
```

Batch evaluation:

```bash
chmod +x eval_modelsel.sh
./eval_modelsel.sh
```

Outputs are saved to `results/`.

## 5) Plot Results

```bash
python perf_plot.py
```

Plots are saved to `results/plots` as PDF files.
