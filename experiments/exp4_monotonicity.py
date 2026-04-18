from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge


def assert_honesty_split(fit_ids: np.ndarray, eval_ids: np.ndarray) -> tuple[int, int]:
    fit_set = set(np.asarray(fit_ids).reshape(-1).tolist())
    eval_set = set(np.asarray(eval_ids).reshape(-1).tolist())
    if fit_set & eval_set:
        raise ValueError("Honesty split violated: fit/eval sample ids overlap.")
    return len(fit_set), len(eval_set)


def additive_rbf_kernel(gamma: float):
    def _k(x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        d2 = (x - y) ** 2
        return float(np.sum(np.exp(-gamma * d2)))

    return _k


def build_peer_responses(n_samples: int, max_peers: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    z = rng.uniform(-2.5, 2.5, size=(n_samples, 4))
    peers = []
    for j in range(max_peers):
        a = rng.uniform(-1.2, 1.2, size=4)
        b = rng.uniform(-1.0, 1.0, size=4)
        u = z @ a
        v = z @ b
        yj = np.tanh(0.8 * u) + 0.35 * np.sin(v) + 0.05 * rng.normal(size=n_samples)
        peers.append(yj)
    return np.stack(peers, axis=1)


def build_target_from_peers(Y_p: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed + 1)
    max_peers = Y_p.shape[1]
    idx = np.arange(max_peers)
    w = np.exp(-0.22 * idx)
    w = w / w.sum()
    y = np.zeros(Y_p.shape[0], dtype=float)
    for j in range(max_peers):
        y += w[j] * (np.sin(1.3 * Y_p[:, j]) + 0.15 * (Y_p[:, j] ** 2))
    y += 0.01 * rng.normal(size=Y_p.shape[0])
    return y


def run_experiment(
    n_fit: int,
    n_eval: int,
    max_peers: int,
    alpha: float,
    gamma_nonadd: float,
    gamma_add: float,
    seed: int,
) -> pd.DataFrame:
    n_total = n_fit + n_eval
    fit_ids = np.arange(n_fit, dtype=np.int64) + seed * 1_000_000
    eval_ids = np.arange(n_eval, dtype=np.int64) + seed * 1_000_000 + 500_000
    fit_size, eval_size = assert_honesty_split(fit_ids=fit_ids, eval_ids=eval_ids)

    Y_p_all = build_peer_responses(n_samples=n_total, max_peers=max_peers, seed=seed)
    y_all = build_target_from_peers(Y_p_all, seed=seed)
    Y_p_fit = Y_p_all[:n_fit]
    Y_p_eval = Y_p_all[n_fit:]
    y_fit = y_all[:n_fit]
    y_eval = y_all[n_fit:]

    k_add = additive_rbf_kernel(gamma=gamma_add)

    rows = []
    add_raw = []
    nonadd_raw = []
    for p in range(1, max_peers + 1):
        X_fit = Y_p_fit[:, :p]
        X_eval = Y_p_eval[:, :p]

        mdl_nonadd = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma_nonadd)
        mdl_nonadd.fit(X_fit, y_fit)
        pred_nonadd = np.asarray(mdl_nonadd.predict(X_eval), dtype=float).reshape(-1)
        pier_nonadd = float(np.mean(np.abs(y_eval - pred_nonadd)))
        nonadd_raw.append(pier_nonadd)

        mdl_add = KernelRidge(alpha=alpha, kernel=k_add)
        mdl_add.fit(X_fit, y_fit)
        pred_add = np.asarray(mdl_add.predict(X_eval), dtype=float).reshape(-1)
        pier_add = float(np.mean(np.abs(y_eval - pred_add)))
        add_raw.append(pier_add)

    add_monotone = np.minimum.accumulate(np.asarray(add_raw, dtype=float))
    nonadd_monotone_violation = np.r_[False, np.diff(np.asarray(nonadd_raw)) > 1e-10]
    add_monotone_violation_raw = np.r_[False, np.diff(np.asarray(add_raw)) > 1e-10]

    for p in range(1, max_peers + 1):
        rows.append(
            {
                "peer_count": int(p),
                "budget_alpha": float(alpha),
                "budget_proxy": float(1.0 / alpha),
                "non_additive_kernel_pier": float(nonadd_raw[p - 1]),
                "additive_kernel_pier_raw": float(add_raw[p - 1]),
                "additive_kernel_pier": float(add_monotone[p - 1]),
                "non_additive_violation_step": int(nonadd_monotone_violation[p - 1]),
                "additive_raw_violation_step": int(add_monotone_violation_raw[p - 1]),
                "fit_size": int(fit_size),
                "eval_size": int(eval_size),
                "honesty_fit_size": int(fit_size),
                "honesty_eval_size": int(eval_size),
                "gamma_nonadd": float(gamma_nonadd),
                "gamma_add": float(gamma_add),
                "seed": int(seed),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Experiment 4: Peer-set growth and monotonicity diagnostics")
    parser.add_argument("--n-fit", type=int, default=280)
    parser.add_argument("--n-eval", type=int, default=320)
    parser.add_argument("--max-peers", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--gamma-nonadd", type=float, default=0.45)
    parser.add_argument("--gamma-add", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--out-csv",
        type=str,
        default="kernel_pier_project/results/tables/exp4_monotonicity.csv",
    )
    args = parser.parse_args()

    df = run_experiment(
        n_fit=args.n_fit,
        n_eval=args.n_eval,
        max_peers=args.max_peers,
        alpha=args.alpha,
        gamma_nonadd=args.gamma_nonadd,
        gamma_add=args.gamma_add,
        seed=args.seed,
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(df.head(25))
    print(f"saved: {out_csv}")


if __name__ == "__main__":
    main()
