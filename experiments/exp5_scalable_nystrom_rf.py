from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable, List

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


def build_peer_matrix(n_samples: int, n_peers: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    z = rng.normal(0.0, 1.0, size=(n_samples, 10))
    Y = np.zeros((n_samples, n_peers), dtype=float)
    for j in range(n_peers):
        a = rng.uniform(-1.2, 1.2, size=10)
        b = rng.uniform(-1.0, 1.0, size=10)
        u = z @ a
        v = z @ b
        Y[:, j] = np.tanh(0.8 * u) + 0.3 * np.sin(v) + 0.05 * rng.normal(size=n_samples)
    return Y


def build_targets(Y_p: np.ndarray, n_targets: int, seed: int) -> List[np.ndarray]:
    rng = np.random.RandomState(seed + 13)
    n_peers = Y_p.shape[1]
    ids = np.arange(n_peers)
    targets = []
    for t in range(n_targets):
        base = np.exp(-(0.12 + 0.03 * t) * ids)
        base = base / base.sum()
        y = np.zeros(Y_p.shape[0], dtype=float)
        nonlin_scale = 0.2 + 0.3 * (t / max(1, n_targets - 1))
        for j in range(n_peers):
            y += base[j] * (Y_p[:, j] + nonlin_scale * np.sin(1.4 * Y_p[:, j]) + 0.05 * (Y_p[:, j] ** 2))
        y += 0.01 * rng.normal(size=Y_p.shape[0])
        targets.append(y)
    return targets


def spearman_rank_corr_desc(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ra = pd.Series(-a).rank(method="average").to_numpy()
    rb = pd.Series(-b).rank(method="average").to_numpy()
    if np.std(ra) <= 1e-12 or np.std(rb) <= 1e-12:
        return 1.0
    return float(np.corrcoef(ra, rb)[0, 1])


def run_experiment(
    n_fit: int,
    n_eval: int,
    n_peers: int,
    n_targets: int,
    lambdas: Iterable[float],
    approx_dims: Iterable[int],
    gamma: float,
    seed: int,
) -> pd.DataFrame:
    n_total = n_fit + n_eval
    fit_ids = np.arange(n_fit, dtype=np.int64) + seed * 1_000_000
    eval_ids = np.arange(n_eval, dtype=np.int64) + seed * 1_000_000 + 500_000
    fit_size, eval_size = assert_honesty_split(fit_ids=fit_ids, eval_ids=eval_ids)

    Y_all = build_peer_matrix(n_samples=n_total, n_peers=n_peers, seed=seed)
    targets_all = build_targets(Y_all, n_targets=n_targets, seed=seed)
    Y_fit = Y_all[:n_fit]
    Y_eval = Y_all[n_fit:]
    target_fit_list = [y[:n_fit] for y in targets_all]
    target_eval_list = [y[n_fit:] for y in targets_all]

    lambdas = [float(v) for v in lambdas]
    approx_dims = [int(d) for d in approx_dims]
    rows = []
    exact_piers = []
    exact_times = []

    for tid in range(n_targets):
        t0 = time.perf_counter()
        exact_res = solve_kernel_residuals(
            y_fit=target_fit_list[tid],
            Y_p_fit=Y_fit,
            y_eval=target_eval_list[tid],
            Y_p_eval=Y_eval,
            lambdas=lambdas,
            kernel_type={"name": "rbf", "gamma": gamma},
            approximation_mode="exact",
        )
        elapsed = float(time.perf_counter() - t0)
        pier = float(np.mean(np.abs(exact_res.best_residuals)))
        exact_piers.append(pier)
        exact_times.append(elapsed)
        rows.append(
            {
                "row_kind": "per_target",
                "target_id": int(tid),
                "method": "exact",
                "approx_dim": np.nan,
                "runtime_sec": elapsed,
                "kernel_pier_estimate": pier,
                "approx_error_abs": 0.0,
                "best_lambda": float(exact_res.best_lambda),
                "fit_size": int(fit_size),
                "eval_size": int(eval_size),
                "honesty_fit_size": int(fit_size),
                "honesty_eval_size": int(eval_size),
                "n_peers": int(n_peers),
                "gamma": float(gamma),
                "seed": int(seed),
            }
        )

    exact_piers_arr = np.asarray(exact_piers, dtype=float)
    rows.append(
        {
            "row_kind": "ranking",
            "target_id": -1,
            "method": "exact",
            "approx_dim": np.nan,
            "runtime_sec": float(np.mean(exact_times)),
            "kernel_pier_estimate": float(np.mean(exact_piers_arr)),
            "approx_error_abs": 0.0,
            "best_lambda": np.nan,
            "fit_size": int(fit_size),
            "eval_size": int(eval_size),
            "honesty_fit_size": int(fit_size),
            "honesty_eval_size": int(eval_size),
            "n_peers": int(n_peers),
            "gamma": float(gamma),
            "seed": int(seed),
            "ranking_spearman_vs_exact": 1.0,
            "top1_match_exact": 1,
        }
    )

    top_exact = int(np.argmax(exact_piers_arr))

    for method in ["nystrom", "rff"]:
        for d in approx_dims:
            approx_piers = []
            runtimes = []
            for tid in range(n_targets):
                t0 = time.perf_counter()
                approx_res = solve_kernel_residuals(
                    y_fit=target_fit_list[tid],
                    Y_p_fit=Y_fit,
                    y_eval=target_eval_list[tid],
                    Y_p_eval=Y_eval,
                    lambdas=lambdas,
                    kernel_type={"name": "rbf", "gamma": gamma},
                    approximation_mode=method,
                    n_components=int(d),
                    random_state=seed,
                )
                elapsed = float(time.perf_counter() - t0)
                pier_hat = float(np.mean(np.abs(approx_res.best_residuals)))
                err = float(abs(pier_hat - exact_piers_arr[tid]))
                approx_piers.append(pier_hat)
                runtimes.append(elapsed)
                rows.append(
                    {
                        "row_kind": "per_target",
                        "target_id": int(tid),
                        "method": method,
                        "approx_dim": int(d),
                        "runtime_sec": elapsed,
                        "kernel_pier_estimate": pier_hat,
                        "approx_error_abs": err,
                        "best_lambda": float(approx_res.best_lambda),
                        "fit_size": int(fit_size),
                        "eval_size": int(eval_size),
                        "honesty_fit_size": int(fit_size),
                        "honesty_eval_size": int(eval_size),
                        "n_peers": int(n_peers),
                        "gamma": float(gamma),
                        "seed": int(seed),
                    }
                )
            approx_piers_arr = np.asarray(approx_piers, dtype=float)
            rank_corr = spearman_rank_corr_desc(approx_piers_arr, exact_piers_arr)
            top_match = int(np.argmax(approx_piers_arr) == top_exact)
            rows.append(
                {
                    "row_kind": "ranking",
                    "target_id": -1,
                    "method": method,
                    "approx_dim": int(d),
                    "runtime_sec": float(np.mean(runtimes)),
                    "kernel_pier_estimate": float(np.mean(approx_piers_arr)),
                    "approx_error_abs": float(np.mean(np.abs(approx_piers_arr - exact_piers_arr))),
                    "best_lambda": np.nan,
                    "fit_size": int(fit_size),
                    "eval_size": int(eval_size),
                    "honesty_fit_size": int(fit_size),
                    "honesty_eval_size": int(eval_size),
                    "n_peers": int(n_peers),
                    "gamma": float(gamma),
                    "seed": int(seed),
                    "ranking_spearman_vs_exact": float(rank_corr),
                    "top1_match_exact": int(top_match),
                }
            )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Experiment 5: scalable approximations for Kernel-PIER")
    parser.add_argument("--n-fit", type=int, default=10000)
    parser.add_argument("--n-eval", type=int, default=2000)
    parser.add_argument("--n-peers", type=int, default=24)
    parser.add_argument("--n-targets", type=int, default=6)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.03])
    parser.add_argument("--approx-dims", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    parser.add_argument("--gamma", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--out-csv",
        type=str,
        default="kernel_pier_project/results/tables/exp5_scalable_approx.csv",
    )
    args = parser.parse_args()

    df = run_experiment(
        n_fit=args.n_fit,
        n_eval=args.n_eval,
        n_peers=args.n_peers,
        n_targets=args.n_targets,
        lambdas=args.lambdas,
        approx_dims=args.approx_dims,
        gamma=args.gamma,
        seed=args.seed,
    )
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(df[df["row_kind"] == "ranking"].sort_values(["method", "approx_dim"]))
    print(f"saved: {out_csv}")


if __name__ == "__main__":
    main()
