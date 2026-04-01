from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kernel_isqed import solve_kernel_residuals


def assert_honesty_split(fit_ids: np.ndarray, eval_ids: np.ndarray) -> tuple[int, int]:
    fit_set = set(np.asarray(fit_ids).reshape(-1).tolist())
    eval_set = set(np.asarray(eval_ids).reshape(-1).tolist())
    if fit_set & eval_set:
        raise ValueError("Honesty split violated: fit/eval sample ids overlap.")
    return len(fit_set), len(eval_set)


def generate_peers(rng: np.random.RandomState, n_fit: int, n_eval: int, n_peers: int) -> tuple[np.ndarray, np.ndarray]:
    Y_p_fit = rng.normal(0.0, 1.0, size=(n_fit, n_peers))
    Y_p_eval = rng.normal(0.0, 1.0, size=(n_eval, n_peers))
    return Y_p_fit, Y_p_eval


def standardize_by_fit(Y_p_fit: np.ndarray, Y_p_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = Y_p_fit.mean(axis=0)
    sigma = Y_p_fit.std(axis=0)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    return (Y_p_fit - mu) / sigma, (Y_p_eval - mu) / sigma


def median_heuristic_gamma(X: np.ndarray) -> float:
    if X.shape[0] < 2:
        return 1.0
    rng = np.random.RandomState(0)
    n = X.shape[0]
    i = rng.randint(0, n, size=20000)
    j = rng.randint(0, n, size=20000)
    mask = i != j
    i = i[mask]
    j = j[mask]
    diffs = X[i] - X[j]
    d2 = np.sum(diffs * diffs, axis=1)
    med = float(np.median(d2))
    if med <= 1e-12:
        return 1.0
    return 1.0 / (2.0 * med)


def build_target(
    rng: np.random.RandomState,
    Y_p_fit: np.ndarray,
    Y_p_eval: np.ndarray,
    complexity: float,
    noise_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    linear_fit = 0.35 * Y_p_fit[:, 0] + 0.20 * Y_p_fit[:, 1]
    linear_eval = 0.35 * Y_p_eval[:, 0] + 0.20 * Y_p_eval[:, 1]

    nonlinear_fit = np.sin(1.5 * Y_p_fit[:, 0]) + 0.5 * (Y_p_fit[:, 2] ** 2)
    nonlinear_eval = np.sin(1.5 * Y_p_eval[:, 0]) + 0.5 * (Y_p_eval[:, 2] ** 2)

    y_fit = (1.0 - complexity) * linear_fit + complexity * nonlinear_fit
    y_eval = (1.0 - complexity) * linear_eval + complexity * nonlinear_eval

    y_fit = y_fit + rng.normal(0.0, noise_std, size=Y_p_fit.shape[0])
    y_eval = y_eval + rng.normal(0.0, noise_std, size=Y_p_eval.shape[0])
    return y_fit, y_eval


def find_min_budget(
    y_fit: np.ndarray,
    X_fit: np.ndarray,
    y_eval: np.ndarray,
    X_eval: np.ndarray,
    lambdas: Iterable[float],
    tau: float,
    gamma: float,
) -> tuple[float, float, float, bool]:
    lambda_list = sorted([float(x) for x in lambdas], reverse=True)
    result = solve_kernel_residuals(
        y_fit=y_fit,
        Y_p_fit=X_fit,
        y_eval=y_eval,
        Y_p_eval=X_eval,
        lambdas=lambda_list,
        kernel_type={"name": "rbf", "gamma": gamma},
    )

    lambda_star = np.nan
    budget_star = np.nan
    pier_star = np.nan
    reachable = False

    for lam in lambda_list:
        pier = float(np.mean(np.abs(result.residuals_by_lambda[lam])))
        if pier <= tau:
            lambda_star = lam
            budget_star = 1.0 / lam
            pier_star = pier
            reachable = True
            break

    return lambda_star, budget_star, pier_star, reachable


def run_experiment(
    seed: int,
    n_targets: int,
    n_fit: int,
    n_eval: int,
    n_peers: int,
    noise_std: float,
    tau: float,
    lambdas: Iterable[float],
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    fit_ids = np.arange(n_fit, dtype=np.int64) + int(seed) * 1_000_000
    eval_ids = np.arange(n_eval, dtype=np.int64) + int(seed) * 1_000_000 + 500_000
    fit_size, eval_size = assert_honesty_split(fit_ids=fit_ids, eval_ids=eval_ids)
    Y_p_fit, Y_p_eval = generate_peers(rng=rng, n_fit=n_fit, n_eval=n_eval, n_peers=n_peers)
    X_fit, X_eval = standardize_by_fit(Y_p_fit, Y_p_eval)
    gamma = median_heuristic_gamma(X_fit)

    complexities = np.linspace(0.05, 1.0, n_targets)
    rows = []
    for idx, c in enumerate(complexities):
        y_fit, y_eval = build_target(
            rng=rng,
            Y_p_fit=Y_p_fit,
            Y_p_eval=Y_p_eval,
            complexity=float(c),
            noise_std=noise_std,
        )
        lambda_star, budget_star, pier_star, reachable = find_min_budget(
            y_fit=y_fit,
            X_fit=X_fit,
            y_eval=y_eval,
            X_eval=X_eval,
            lambdas=lambdas,
            tau=tau,
            gamma=gamma,
        )
        rows.append(
            {
                "target_id": int(idx),
                "true_complexity": float(c),
                "tau": float(tau),
                "lambda_star": float(lambda_star) if reachable else np.nan,
                "B_tau": float(budget_star) if reachable else np.nan,
                "pier_at_B_tau": float(pier_star) if reachable else np.nan,
                "reachable": int(reachable),
                "gamma": float(gamma),
                "n_fit": int(n_fit),
                "n_eval": int(n_eval),
                "n_peers": int(n_peers),
                "honesty_fit_size": int(fit_size),
                "honesty_eval_size": int(eval_size),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Experiment 2: Replacement complexity as a summary statistic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-targets", type=int, default=20)
    parser.add_argument("--n-fit", type=int, default=260)
    parser.add_argument("--n-eval", type=int, default=360)
    parser.add_argument("--n-peers", type=int, default=8)
    parser.add_argument("--noise-std", type=float, default=0.02)
    parser.add_argument("--tau", type=float, default=0.20)
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6],
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="kernel_pier_project/results/tables/exp2_replacement_complexity.csv",
    )
    args = parser.parse_args()

    df = run_experiment(
        seed=args.seed,
        n_targets=args.n_targets,
        n_fit=args.n_fit,
        n_eval=args.n_eval,
        n_peers=args.n_peers,
        noise_std=args.noise_std,
        tau=args.tau,
        lambdas=args.lambdas,
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(df)
    print(f"saved: {out_csv}")


if __name__ == "__main__":
    main()
