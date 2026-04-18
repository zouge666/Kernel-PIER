from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.kernel_ridge import KernelRidge

ROOT_DIR = Path(__file__).resolve().parents[2]
KERNEL_PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(KERNEL_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(KERNEL_PROJECT_DIR))

from isqed.geometry import DISCOSolver
from isqed.real_world import AdversarialFGSMIntervention
from exp3b_imagenet_kernel_adv import (
    assert_honesty_split,
    build_fallback_models,
    build_synthetic_samples,
    load_real_imagefolder_samples,
    load_real_models,
    median_heuristic_gamma,
    split_fit_eval,
    standardize_by_fit,
)


def _fit_predict_kernel_ridge(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_eval: np.ndarray,
    kernel_type: str,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    params = {}
    if kernel_type == "rbf":
        params["gamma"] = float(gamma)
    model = KernelRidge(alpha=float(lam), kernel=kernel_type, **params)
    model.fit(x_fit, y_fit)
    pred_fit = np.asarray(model.predict(x_fit), dtype=float).reshape(-1)
    pred_eval = np.asarray(model.predict(x_eval), dtype=float).reshape(-1)
    return pred_fit, pred_eval


def run_experiment(
    data_root: str,
    max_samples: int,
    sample_seed: int,
    dose_epsilon: float,
    lambdas: Iterable[float],
    kernel_type: str,
    gamma: Optional[float],
    monotonic_tol: float,
) -> pd.DataFrame:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    samples = load_real_imagefolder_samples(data_root=data_root, max_samples=max_samples, seed=sample_seed)
    backend_data = "imagefolder"
    if samples is None:
        samples = build_synthetic_samples(max_samples=max_samples, seed=sample_seed)
        backend_data = "synthetic"

    models = None
    backend_model = "fallback_tiny"
    if backend_data == "imagefolder":
        models = load_real_models(device=device)
        backend_model = "torchvision"
    if models is None:
        models = build_fallback_models(device=device)
        backend_model = "fallback_tiny"

    fit_samples, eval_samples, fit_ids, eval_ids = split_fit_eval(samples=samples, seed=sample_seed, fit_frac=0.5)
    fit_size, eval_size = assert_honesty_split(fit_ids=fit_ids, eval_ids=eval_ids)

    ref_model = None
    for m in models:
        if "ResNet50" in m.name and "Robust" not in m.name:
            ref_model = m.model
            break
    if ref_model is None:
        ref_model = models[0].model

    adv = AdversarialFGSMIntervention(ref_model=ref_model, device=device)
    lambda_list = sorted([float(v) for v in lambdas], reverse=True)

    rows = []
    for t_idx, target in enumerate(models):
        peers = [m for i, m in enumerate(models) if i != t_idx]

        y_fit_list = []
        y_peer_fit_list = []
        for sample in fit_samples:
            adv_sample = adv.apply(sample, epsilon=float(dose_epsilon))
            y_t = target._forward(adv_sample)
            y_p = [p._forward(adv_sample) for p in peers]
            y_fit_list.append(float(y_t))
            y_peer_fit_list.append([float(v) for v in y_p])
        y_fit = np.asarray(y_fit_list, dtype=float)
        y_peer_fit = np.asarray(y_peer_fit_list, dtype=float)

        y_eval_list = []
        y_peer_eval_list = []
        for sample in eval_samples:
            adv_sample = adv.apply(sample, epsilon=float(dose_epsilon))
            y_t = target._forward(adv_sample)
            y_p = [p._forward(adv_sample) for p in peers]
            y_eval_list.append(float(y_t))
            y_peer_eval_list.append([float(v) for v in y_p])
        y_eval = np.asarray(y_eval_list, dtype=float)
        y_peer_eval = np.asarray(y_peer_eval_list, dtype=float)

        dist_fit, w_hat = DISCOSolver.solve_weights_and_distance(y_fit.reshape(-1, 1), y_peer_fit)
        w_hat = np.asarray(w_hat, dtype=float).reshape(-1)

        y_mix_fit = y_peer_fit @ w_hat
        y_mix_eval = y_peer_eval @ w_hat
        convex_pier_fit = float(np.mean(np.abs(y_fit - y_mix_fit)))
        convex_pier_eval = float(np.mean(np.abs(y_eval - y_mix_eval)))

        x_fit_k, x_eval_k = standardize_by_fit(y_peer_fit, y_peer_eval)
        gamma_used = median_heuristic_gamma(x_fit_k) if gamma is None else float(gamma)

        for lam in lambda_list:
            pred_fit, pred_eval = _fit_predict_kernel_ridge(
                x_fit=x_fit_k,
                y_fit=y_fit,
                x_eval=x_eval_k,
                kernel_type=kernel_type,
                gamma=float(gamma_used),
                lam=float(lam),
            )
            fit_res = y_fit - pred_fit
            eval_res = y_eval - pred_eval
            kernel_pier_fit = float(np.mean(np.abs(fit_res)))
            kernel_pier_eval = float(np.mean(np.abs(eval_res)))
            overfit_gap = float(kernel_pier_eval - kernel_pier_fit)

            rows.append(
                {
                    "target_model": target.name,
                    "target_group": "Robust" if "Robust" in target.name else "Standard",
                    "dose_epsilon": float(dose_epsilon),
                    "lambda": float(lam),
                    "budget": float(1.0 / float(lam)),
                    "convex_fit_distance": float(dist_fit),
                    "convex_pier_eval": convex_pier_eval,
                    "convex_pier_fit": convex_pier_fit,
                    "kernel_pier_eval": kernel_pier_eval,
                    "kernel_pier_fit": kernel_pier_fit,
                    "overfit_gap": overfit_gap,
                    "pier_drop_eval": float(convex_pier_eval - kernel_pier_eval),
                    "pier_drop_fit": float(convex_pier_fit - kernel_pier_fit),
                    "n_fit": int(len(fit_samples)),
                    "n_eval": int(len(eval_samples)),
                    "honesty_fit_size": int(fit_size),
                    "honesty_eval_size": int(eval_size),
                    "n_models_total": int(len(models)),
                    "n_peers": int(len(peers)),
                    "kernel_type": kernel_type,
                    "gamma": float(gamma_used),
                    "data_backend": backend_data,
                    "model_backend": backend_model,
                    "monotonic_tol": float(monotonic_tol),
                }
            )

    df = pd.DataFrame(rows)
    df = df.sort_values(["target_model", "budget"], ascending=[True, True]).reset_index(drop=True)

    df["eval_delta_vs_prev"] = df.groupby("target_model", sort=False)["kernel_pier_eval"].diff()
    df["fit_delta_vs_prev"] = df.groupby("target_model", sort=False)["kernel_pier_fit"].diff()
    df["eval_increase_vs_prev"] = (df["eval_delta_vs_prev"] > float(monotonic_tol)).astype(int)
    df["fit_increase_vs_prev"] = (df["fit_delta_vs_prev"] > float(monotonic_tol)).astype(int)

    eval_rise_count = df.groupby("target_model", as_index=False)["eval_increase_vs_prev"].sum().rename(
        columns={"eval_increase_vs_prev": "eval_rise_count"}
    )
    fit_rise_count = df.groupby("target_model", as_index=False)["fit_increase_vs_prev"].sum().rename(
        columns={"fit_increase_vs_prev": "fit_rise_count"}
    )
    df = df.merge(eval_rise_count, on="target_model", how="left")
    df = df.merge(fit_rise_count, on="target_model", how="left")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp3a2: CV lambda-overfitting audit")
    parser.add_argument("--data-root", type=str, default="./data/imagenet/val")
    parser.add_argument("--max-samples", type=int, default=80)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--dose-epsilon", type=float, default=0.1)
    parser.add_argument("--kernel-type", type=str, default="rbf")
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--monotonic-tol", type=float, default=1e-4)
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[3.0, 1.0, 3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5],
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="results/tables/exp3a2_cv_lambda_overfit.csv",
    )
    args = parser.parse_args()

    df = run_experiment(
        data_root=args.data_root,
        max_samples=args.max_samples,
        sample_seed=args.sample_seed,
        dose_epsilon=args.dose_epsilon,
        lambdas=args.lambdas,
        kernel_type=args.kernel_type,
        gamma=args.gamma,
        monotonic_tol=args.monotonic_tol,
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    summary = (
        df.groupby(["target_model", "target_group"], as_index=False)
        .agg(
            convex_eval=("convex_pier_eval", "mean"),
            best_kernel_eval=("kernel_pier_eval", "min"),
            mean_overfit_gap=("overfit_gap", "mean"),
            eval_rise_count=("eval_rise_count", "max"),
        )
        .sort_values(["target_group", "target_model"])
    )
    print(summary)
    print(f"saved: {out_csv}")


if __name__ == "__main__":
    main()
