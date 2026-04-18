from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge

try:
    import torch  # type: ignore
except Exception:
    torch = None

try:
    from datasets import load_dataset  # type: ignore
except Exception:
    load_dataset = None

ROOT_DIR = Path(__file__).resolve().parents[2]
KERNEL_PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(KERNEL_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(KERNEL_PROJECT_DIR))

from isqed.core import Intervention, ModelUnit
from isqed.ecosystem import Ecosystem

try:
    from isqed.real_world import HuggingFaceWrapper
except Exception:
    HuggingFaceWrapper = None

from experiments.utils import make_stable_seed

try:
    from isqed.geometry import DISCOSolver
except Exception:
    DISCOSolver = None


class DeterministicTextModel(ModelUnit):
    def __init__(self, name: str, a: float, b: float, c: float):
        super().__init__(name=name)
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)

    def _forward(self, text_input):
        text = str(text_input)
        tokens = text.split()
        n = max(1, len(tokens))
        n_mask = sum(1 for t in tokens if t == "[MASK]")
        mask_ratio = n_mask / float(n)
        digest = hashlib.md5((self.name + "||" + text).encode("utf-8")).hexdigest()
        h = (int(digest[:8], 16) % 10000) / 10000.0
        z = self.a * (1.0 - mask_ratio) + self.b * np.sin(4.0 * h + self.c) + 0.2 * np.tanh(2.0 * h - 1.0)
        return float(1.0 / (1.0 + np.exp(-z)))


class SimpleMaskingIntervention(Intervention):
    def apply(self, x, theta: float, seed: int | None = None):
        text = str(x)
        tokens = text.split()
        if not tokens:
            return text
        frac = min(max(float(theta), 0.0), 1.0)
        k = int(round(frac * len(tokens)))
        if k <= 0:
            return text
        rng = np.random.RandomState(0 if seed is None else int(seed))
        idx = np.arange(len(tokens))
        rng.shuffle(idx)
        mask_idx = set(idx[:k].tolist())
        out = ["[MASK]" if i in mask_idx else t for i, t in enumerate(tokens)]
        return " ".join(out)


def assert_honesty_split(fit_ids: np.ndarray, eval_ids: np.ndarray) -> tuple[int, int]:
    fit_set = set(np.asarray(fit_ids).reshape(-1).tolist())
    eval_set = set(np.asarray(eval_ids).reshape(-1).tolist())
    if fit_set & eval_set:
        raise ValueError("Honesty split violated: fit/eval sample ids overlap.")
    return len(fit_set), len(eval_set)


def standardize_by_fit(y_p_fit: np.ndarray, y_p_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = y_p_fit.mean(axis=0)
    sigma = y_p_fit.std(axis=0)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    return (y_p_fit - mu) / sigma, (y_p_eval - mu) / sigma


def median_heuristic_gamma(x: np.ndarray) -> float:
    if x.shape[0] < 2:
        return 1.0
    idx = np.arange(x.shape[0])
    pairs = np.stack(np.meshgrid(idx, idx), axis=-1).reshape(-1, 2)
    pairs = pairs[pairs[:, 0] < pairs[:, 1]]
    if pairs.shape[0] > 15000:
        rng = np.random.RandomState(0)
        sel = rng.choice(pairs.shape[0], size=15000, replace=False)
        pairs = pairs[sel]
    diffs = x[pairs[:, 0]] - x[pairs[:, 1]]
    d2 = np.sum(diffs * diffs, axis=1)
    med = float(np.median(d2))
    if med <= 1e-12:
        return 1.0
    return 1.0 / (2.0 * med)


def build_fallback_models():
    peers = [
        DeterministicTextModel("peer-bert", 3.2, 1.1, 0.1),
        DeterministicTextModel("peer-distilbert", 2.8, 0.9, 0.5),
        DeterministicTextModel("peer-albert", 2.4, 1.3, 1.0),
        DeterministicTextModel("peer-xlnet", 2.2, 1.0, 1.5),
    ]
    targets = [
        {"model": DeterministicTextModel("target-roberta", 1.9, 1.8, 0.9), "name": "Architectural Divergence (RoBERTa)", "type": "Low Redundancy"},
        {"model": peers[1], "name": "Perfect Redundancy (Clone)", "type": "High Redundancy", "clone_of_peer_idx": 1},
        {"model": DeterministicTextModel("target-distil-variant", 2.6, 1.2, 0.3), "name": "Parametric Divergence (Finetuned)", "type": "Uniqueness"},
    ]
    return peers, targets


def build_hf_models(device: str):
    if HuggingFaceWrapper is None:
        return [], []

    peer_ids = [
        "textattack/bert-base-uncased-SST-2",
        "textattack/distilbert-base-uncased-SST-2",
        "textattack/albert-base-v2-SST-2",
        "textattack/xlnet-base-cased-SST-2",
    ]
    peers = []
    for pid in peer_ids:
        try:
            peers.append(HuggingFaceWrapper(pid, device))
        except Exception:
            continue

    targets = []
    try:
        targets.append(
            {
                "model": HuggingFaceWrapper("textattack/roberta-base-SST-2", device),
                "name": "Architectural Divergence (RoBERTa)",
                "type": "Low Redundancy",
            }
        )
    except Exception:
        pass

    distil_ref = next((p for p in peers if "distilbert" in p.name.lower()), None)
    if distil_ref is not None:
        targets.append(
            {
                "model": distil_ref,
                "name": "Perfect Redundancy (Clone)",
                "type": "High Redundancy",
                "clone_of_peer_idx": peers.index(distil_ref),
            }
        )
    try:
        targets.append(
            {
                "model": HuggingFaceWrapper("distilbert-base-uncased-finetuned-sst-2-english", device),
                "name": "Parametric Divergence (Finetuned)",
                "type": "Uniqueness",
            }
        )
    except Exception:
        pass
    return peers, targets


def load_sst2_sentences(max_samples: int, seed: int):
    sentences = None
    if load_dataset is not None:
        try:
            ds = load_dataset("glue", "sst2", split="validation")
            sentences = ds["sentence"][:max_samples]
        except Exception:
            sentences = None
    if sentences is None:
        base = [
            "This movie is great.",
            "Terrible acting and boring scenes.",
            "I loved every second of this film.",
            "The storyline felt weak and predictable.",
            "Excellent performance and direction.",
            "I would not recommend this to anyone.",
        ]
        repeats = max(1, (max_samples + len(base) - 1) // len(base))
        sentences = (base * repeats)[:max_samples]

    arr = np.asarray(sentences, dtype=object)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(arr))
    arr = arr[perm]
    n_fit = len(arr) // 2
    fit_texts = arr[:n_fit].tolist()
    eval_texts = arr[n_fit:].tolist()
    fit_ids = perm[:n_fit]
    eval_ids = perm[n_fit:]
    return fit_texts, eval_texts, fit_ids, eval_ids


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


def _project_to_simplex(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=float).reshape(-1)
    if x.size == 0:
        return x
    u = np.sort(x)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, len(u) + 1)
    cond = u - cssv / ind > 0
    if not np.any(cond):
        return np.full_like(x, 1.0 / max(1, x.size))
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(x - theta, 0.0)
    s = w.sum()
    if s <= 0:
        return np.full_like(x, 1.0 / max(1, x.size))
    return w / s


def _convex_weights_and_pier(
    y_fit: np.ndarray, y_p_fit: np.ndarray, y_eval: np.ndarray, y_p_eval: np.ndarray
) -> tuple[float, float]:
    if DISCOSolver is not None:
        dist_fit, w_hat = DISCOSolver.solve_weights_and_distance(y_fit.reshape(-1, 1), y_p_fit)
        w_hat = np.asarray(w_hat, dtype=float).reshape(-1)
        y_mix_eval = y_p_eval @ w_hat
        convex_residual_eval = np.abs(y_eval - y_mix_eval)
        return float(dist_fit), float(np.mean(convex_residual_eval))

    # Fallback without cvxpy: unconstrained least squares + simplex projection.
    w_ls, *_ = np.linalg.lstsq(np.asarray(y_p_fit, dtype=float), np.asarray(y_fit, dtype=float), rcond=None)
    w_hat = _project_to_simplex(w_ls)
    y_mix_fit = y_p_fit @ w_hat
    y_mix_eval = y_p_eval @ w_hat
    dist_fit = float(np.mean(np.abs(np.asarray(y_fit, dtype=float) - y_mix_fit)))
    convex_pier = float(np.mean(np.abs(np.asarray(y_eval, dtype=float) - y_mix_eval)))
    return dist_fit, convex_pier


def run_bert_lambda_overfit_audit(
    theta: float,
    lambdas: Iterable[float],
    max_samples: int,
    data_seed: int,
    kernel_type: str,
    gamma: float | None,
    monotonic_tol: float,
) -> pd.DataFrame:
    device = "cpu"
    if torch is not None and torch.cuda.is_available():
        device = "cuda"
        torch.manual_seed(0)

    peers, targets = build_hf_models(device=device)
    use_fallback = False
    if len(peers) < 2 or len(targets) == 0:
        peers, targets = build_fallback_models()
        use_fallback = True

    fit_texts, eval_texts, fit_ids, eval_ids = load_sst2_sentences(max_samples=max_samples, seed=data_seed)
    fit_size, eval_size = assert_honesty_split(fit_ids=fit_ids, eval_ids=eval_ids)

    intervention = SimpleMaskingIntervention()
    lambda_list = sorted([float(v) for v in lambdas], reverse=True)

    rows = []
    for t_info in targets:
        eco = Ecosystem(target=t_info["model"], peers=peers)

        fit_x = fit_texts
        fit_theta = [float(theta)] * len(fit_x)
        fit_seeds = [int(make_stable_seed(text=x, theta=float(theta))) for x in fit_x]
        y_fit, y_p_fit = eco.batched_query(X=fit_x, Thetas=fit_theta, intervention=intervention, seeds=fit_seeds)

        eval_x = eval_texts
        eval_theta = [float(theta)] * len(eval_x)
        eval_seeds = [int(make_stable_seed(text=x, theta=float(theta))) for x in eval_x]
        y_eval, y_p_eval = eco.batched_query(X=eval_x, Thetas=eval_theta, intervention=intervention, seeds=eval_seeds)

        dist_fit, convex_pier = _convex_weights_and_pier(
            y_fit=np.asarray(y_fit, dtype=float).reshape(-1),
            y_p_fit=np.asarray(y_p_fit, dtype=float),
            y_eval=np.asarray(y_eval, dtype=float).reshape(-1),
            y_p_eval=np.asarray(y_p_eval, dtype=float),
        )

        x_fit_k, x_eval_k = standardize_by_fit(y_p_fit, y_p_eval)
        gamma_used = median_heuristic_gamma(x_fit_k) if gamma is None else float(gamma)

        for lam in lambda_list:
            pred_fit, pred_eval = _fit_predict_kernel_ridge(
                x_fit=x_fit_k,
                y_fit=np.asarray(y_fit, dtype=float).reshape(-1),
                x_eval=x_eval_k,
                kernel_type=kernel_type,
                gamma=float(gamma_used),
                lam=float(lam),
            )
            y_fit_vec = np.asarray(y_fit, dtype=float).reshape(-1)
            y_eval_vec = np.asarray(y_eval, dtype=float).reshape(-1)
            fit_res = y_fit_vec - pred_fit
            eval_res = y_eval_vec - pred_eval

            kernel_pier_fit = float(np.mean(np.abs(fit_res)))
            kernel_pier_eval = float(np.mean(np.abs(eval_res)))
            overfit_gap = float(kernel_pier_eval - kernel_pier_fit)

            rows.append(
                {
                    "target_model": t_info["name"],
                    "target_group": t_info["type"],
                    "theta": float(theta),
                    "lambda": float(lam),
                    "budget": float(1.0 / float(lam)),
                    "convex_fit_distance": float(dist_fit),
                    "convex_pier": convex_pier,
                    "kernel_pier_eval": kernel_pier_eval,
                    "kernel_pier_fit": kernel_pier_fit,
                    "overfit_gap": overfit_gap,
                    "pier_drop_eval": float(convex_pier - kernel_pier_eval),
                    "pier_drop_fit": float(convex_pier - kernel_pier_fit),
                    "num_peers": int(len(peers)),
                    "n_fit": int(len(fit_texts)),
                    "n_eval": int(len(eval_texts)),
                    "honesty_fit_size": int(fit_size),
                    "honesty_eval_size": int(eval_size),
                    "kernel_type": str(kernel_type),
                    "gamma": float(gamma_used),
                    "model_backend": "fallback" if use_fallback else "huggingface",
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
    parser = argparse.ArgumentParser(description="Exp3a1: Lambda-overfitting audit for BERT kernel substitution")
    parser.add_argument("--theta", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--data-seed", type=int, default=0)
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
        default="results/tables/exp3a1_bert_lambda_overfit.csv",
    )
    args = parser.parse_args()

    df = run_bert_lambda_overfit_audit(
        theta=args.theta,
        lambdas=args.lambdas,
        max_samples=args.max_samples,
        data_seed=args.data_seed,
        kernel_type=args.kernel_type,
        gamma=args.gamma,
        monotonic_tol=args.monotonic_tol,
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    summary = (
        df.groupby("target_model", as_index=False)
        .agg(
            convex_pier=("convex_pier", "mean"),
            best_kernel_eval=("kernel_pier_eval", "min"),
            mean_overfit_gap=("overfit_gap", "mean"),
            eval_rise_count=("eval_rise_count", "max"),
        )
    )
    print(summary)
    print(f"saved: {out_csv}")


if __name__ == "__main__":
    main()
