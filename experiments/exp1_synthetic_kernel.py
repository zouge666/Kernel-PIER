from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import cvxpy as cp
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


def generate_peer_matrices(
    rng: np.random.RandomState,
    n_fit: int,
    n_eval: int,
    n_peers: int,
) -> tuple[np.ndarray, np.ndarray]:
    Y_p_fit = rng.normal(loc=0.0, scale=1.0, size=(n_fit, n_peers))
    Y_p_eval = rng.normal(loc=0.0, scale=1.0, size=(n_eval, n_peers))
    return Y_p_fit, Y_p_eval


def condition_convex(
    rng: np.random.RandomState,
    Y_p_fit: np.ndarray,
    Y_p_eval: np.ndarray,
    noise_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_peers = Y_p_fit.shape[1]
    w_true = rng.dirichlet(np.ones(n_peers))
    y_fit = Y_p_fit @ w_true + rng.normal(0.0, noise_std, size=Y_p_fit.shape[0])
    y_eval = Y_p_eval @ w_true + rng.normal(0.0, noise_std, size=Y_p_eval.shape[0])
    return y_fit, y_eval


def condition_nonlinear(
    rng: np.random.RandomState,
    Y_p_fit: np.ndarray,
    Y_p_eval: np.ndarray,
    noise_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    y_fit = (
        np.sin(1.4 * Y_p_fit[:, 0])
        + 0.5 * Y_p_fit[:, 1]
        + 0.2 * (Y_p_fit[:, 2] ** 2)
        + rng.normal(0.0, noise_std, size=Y_p_fit.shape[0])
    )
    y_eval = (
        np.sin(1.4 * Y_p_eval[:, 0])
        + 0.5 * Y_p_eval[:, 1]
        + 0.2 * (Y_p_eval[:, 2] ** 2)
        + rng.normal(0.0, noise_std, size=Y_p_eval.shape[0])
    )
    return y_fit, y_eval


def standardize_by_fit(Y_p_fit: np.ndarray, Y_p_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = Y_p_fit.mean(axis=0)
    sigma = Y_p_fit.std(axis=0)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    return (Y_p_fit - mu) / sigma, (Y_p_eval - mu) / sigma


def median_heuristic_gamma(X: np.ndarray) -> float:
    if X.shape[0] < 2:
        return 1.0
    idx = np.arange(X.shape[0])
    pairs = np.stack(np.meshgrid(idx, idx), axis=-1).reshape(-1, 2)
    pairs = pairs[pairs[:, 0] < pairs[:, 1]]
    if pairs.shape[0] > 20000:
        rng = np.random.RandomState(0)
        sel = rng.choice(pairs.shape[0], size=20000, replace=False)
        pairs = pairs[sel]
    diffs = X[pairs[:, 0]] - X[pairs[:, 1]]
    d2 = np.sum(diffs * diffs, axis=1)
    med = float(np.median(d2))
    if med <= 1e-12:
        return 1.0
    return 1.0 / (2.0 * med)


def solve_convex_disco(y_fit: np.ndarray, Y_p_fit: np.ndarray) -> np.ndarray:
    n_peers = Y_p_fit.shape[1]
    w = cp.Variable(n_peers)
    objective = cp.Minimize(cp.norm(y_fit - Y_p_fit @ w, 2))
    constraints = [w >= 0, cp.sum(w) == 1]
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.ECOS)
    except Exception:
        try:
            problem.solve(solver=cp.SCS)
        except Exception:
            problem.solve()
    if w.value is None:
        return np.ones(n_peers, dtype=float) / n_peers
    return np.asarray(w.value, dtype=float).reshape(-1)


def evaluate_condition(
    seed: int,
    condition_name: str,
    fit_size: int,
    eval_size: int,
    y_fit: np.ndarray,
    Y_p_fit: np.ndarray,
    y_eval: np.ndarray,
    Y_p_eval: np.ndarray,
    lambdas: Iterable[float],
    kernel_type: str,
    gamma: float | None,
) -> list[dict]:
    lambda_list = [float(x) for x in lambdas]

    w_convex = solve_convex_disco(y_fit=y_fit, Y_p_fit=Y_p_fit)
    y_pred_convex = Y_p_eval @ w_convex
    residual_convex = y_eval - y_pred_convex
    pier_convex = float(np.mean(np.abs(residual_convex)))

    X_fit_k, X_eval_k = standardize_by_fit(Y_p_fit, Y_p_eval)
    gamma_used = median_heuristic_gamma(X_fit_k) if gamma is None else float(gamma)

    kernel_result = solve_kernel_residuals(
        y_fit=y_fit,
        Y_p_fit=X_fit_k,
        y_eval=y_eval,
        Y_p_eval=X_eval_k,
        lambdas=lambda_list,
        kernel_type={"name": kernel_type, "gamma": gamma_used},
    )

    rows = []
    for lam in lambda_list:
        budget = 1.0 / lam
        rows.append(
            {
                "seed": int(seed),
                "condition": condition_name,
                "method": "ConvexDISCO",
                "lambda": float(lam),
                "budget": float(budget),
                "pier_score": pier_convex,
                "n_fit": int(Y_p_fit.shape[0]),
                "n_eval": int(Y_p_eval.shape[0]),
                "n_peers": int(Y_p_fit.shape[1]),
                "kernel_type": kernel_type,
                "gamma": float(gamma_used),
                "honesty_fit_size": int(fit_size),
                "honesty_eval_size": int(eval_size),
            }
        )

        res = kernel_result.residuals_by_lambda[float(lam)]
        pier_kernel = float(np.mean(np.abs(res)))
        rows.append(
            {
                "seed": int(seed),
                "condition": condition_name,
                "method": "KernelDISCO",
                "lambda": float(lam),
                "budget": float(budget),
                "pier_score": pier_kernel,
                "n_fit": int(Y_p_fit.shape[0]),
                "n_eval": int(Y_p_eval.shape[0]),
                "n_peers": int(Y_p_fit.shape[1]),
                "kernel_type": kernel_type,
                "gamma": float(gamma_used),
                "honesty_fit_size": int(fit_size),
                "honesty_eval_size": int(eval_size),
            }
        )
    return rows


def run_experiment(
    seeds: Iterable[int],
    n_fit: int,
    n_eval: int,
    n_peers: int,
    convex_noise_std: float,
    nonlinear_noise_std: float,
    lambdas: Iterable[float],
    kernel_type: str,
    gamma: float | None,
) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        rng = np.random.RandomState(int(seed))
        fit_ids = np.arange(n_fit, dtype=np.int64) + int(seed) * 1_000_000
        eval_ids = np.arange(n_eval, dtype=np.int64) + int(seed) * 1_000_000 + 500_000
        fit_size, eval_size = assert_honesty_split(fit_ids=fit_ids, eval_ids=eval_ids)
        Y_p_fit, Y_p_eval = generate_peer_matrices(
            rng=rng,
            n_fit=n_fit,
            n_eval=n_eval,
            n_peers=n_peers,
        )

        y_fit_a, y_eval_a = condition_convex(
            rng=rng,
            Y_p_fit=Y_p_fit,
            Y_p_eval=Y_p_eval,
            noise_std=convex_noise_std,
        )
        rows.extend(
            evaluate_condition(
                seed=int(seed),
                condition_name="A_ConvexHull",
                fit_size=fit_size,
                eval_size=eval_size,
                y_fit=y_fit_a,
                Y_p_fit=Y_p_fit,
                y_eval=y_eval_a,
                Y_p_eval=Y_p_eval,
                lambdas=lambdas,
                kernel_type=kernel_type,
                gamma=gamma,
            )
        )

        y_fit_b, y_eval_b = condition_nonlinear(
            rng=rng,
            Y_p_fit=Y_p_fit,
            Y_p_eval=Y_p_eval,
            noise_std=nonlinear_noise_std,
        )
        rows.extend(
            evaluate_condition(
                seed=int(seed),
                condition_name="B_NonlinearRKHS",
                fit_size=fit_size,
                eval_size=eval_size,
                y_fit=y_fit_b,
                Y_p_fit=Y_p_fit,
                y_eval=y_eval_b,
                Y_p_eval=Y_p_eval,
                lambdas=lambdas,
                kernel_type=kernel_type,
                gamma=gamma,
            )
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Controlled synthetic ecosystems for Kernel-PIER")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument("--n-fit", type=int, default=240)
    parser.add_argument("--n-eval", type=int, default=320)
    parser.add_argument("--n-peers", type=int, default=8)
    parser.add_argument("--convex-noise-std", type=float, default=0.003)
    parser.add_argument("--nonlinear-noise-std", type=float, default=0.02)
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
    )
    parser.add_argument("--kernel-type", type=str, default="rbf")
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument(
        "--out-csv",
        type=str,
        default="../results/tables/exp1_synthetic_kernel.csv",
    )
    args = parser.parse_args()

    df = run_experiment(
        seeds=args.seeds,
        n_fit=args.n_fit,
        n_eval=args.n_eval,
        n_peers=args.n_peers,
        convex_noise_std=args.convex_noise_std,
        nonlinear_noise_std=args.nonlinear_noise_std,
        lambdas=args.lambdas,
        kernel_type=args.kernel_type,
        gamma=args.gamma,
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    summary = (
        df.groupby(["condition", "method", "lambda", "budget"], as_index=False)["pier_score"]
        .mean()
        .sort_values(["condition", "method", "lambda"])
    )
    print(summary)
    print(f"saved: {out_csv}")


if __name__ == "__main__":
    main()
